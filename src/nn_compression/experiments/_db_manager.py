import pandas as pd
from typing import Any, Dict, Optional, Type, TypeVar
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    DateTime,
    UniqueConstraint,
)
from ._models import Experiment
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from datetime import datetime
import uuid
import os

# Type variable for ORM models
T = TypeVar("T")

Base = declarative_base()


class CalibrationConfigSql(Base):
    __tablename__ = "calibration_configs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset = Column(String, nullable=False)
    nbatches = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    __table_args__ = (UniqueConstraint("dataset", "nbatches", "batch_size"),)


class CompressionConfigSql(Base):
    __tablename__ = "compression_configs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    method = Column(String, nullable=False)
    log10_lm = Column(Float, nullable=False)
    groupsize = Column(Integer, nullable=False)
    entropy_model = Column(String, nullable=False)
    nbins = Column(Integer, nullable=False)
    scan_order_major = Column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "method",
            "log10_lm",
            "groupsize",
            "entropy_model",
            "nbins",
            "scan_order_major",
        ),
    )


class EvaluationConfigSql(Base):
    __tablename__ = "evaluation_configs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset = Column(String, nullable=False)
    nbatches = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    entropy_model = Column(String, nullable=False)
    transpose = Column(Boolean, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "dataset", "nbatches", "batch_size", "entropy_model", "transpose"
        ),
    )


class NetworkConfigSql(Base):
    __tablename__ = "network_configs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    train_dataset = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint("name", "train_dataset"),)


class ExperimentResultSql(Base):
    __tablename__ = "experiment_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    performance = Column(Float)
    bitrate = Column(Float)


class ExperimentSql(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_name = Column(String, nullable=False)
    experiment_description = Column(String)
    experiment_task = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    calibration_config_id = Column(Integer, ForeignKey("calibration_configs.id"))
    compression_config_id = Column(Integer, ForeignKey("compression_configs.id"))
    evaluation_config_id = Column(Integer, ForeignKey("evaluation_configs.id"))
    network_config_id = Column(Integer, ForeignKey("network_configs.id"))
    results_id = Column(Integer, ForeignKey("experiment_results.id"))
    calibration_config = relationship("CalibrationConfigSql")
    compression_config = relationship("CompressionConfigSql")
    evaluation_config = relationship("EvaluationConfigSql")
    network_config = relationship("NetworkConfigSql")
    results = relationship("ExperimentResultSql")


class DatabaseManager:
    """
    Manages database operations for experiments and related configurations.

    Handles the persistence of experiment data, creating and retrieving
    database records while maintaining the separation between domain models
    and SQLAlchemy ORM models.
    """

    def __init__(self, db_path: str = "experiments.db") -> None:
        """
        Initialize the database manager.

        Args:
            db_path: Optional path to the SQLite database file.
                     If None, uses the default path from get_db_path().
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_experiment(self, exp: Experiment) -> None:
        """
        Save an experiment from the domain model to the database.

        Takes a domain Experiment object and persists it to the database by
        creating the appropriate SQLAlchemy model instances.

        Args:
            exp: The domain Experiment object to save.

        Raises:
            Exception: If there is an error during the save operation.
                      The transaction will be rolled back.
        """
        session = self.Session()
        try:
            results = ExperimentResultSql(
                performance=exp.output_results.performance,
                bitrate=exp.output_results.bitrate,
            )
            session.add(results)
            session.commit()

            experiment = ExperimentSql(
                experiment_name=exp.config.get("experiment_name", ""),
                experiment_description=exp.config.get("experiment_description", ""),
                experiment_task=exp.config.get("experiment_task", ""),
                calibration_config=self.get_or_create(
                    session, CalibrationConfigSql, exp.config.get("calibration", {})
                ),
                compression_config=self.get_or_create(
                    session, CompressionConfigSql, exp.config.get("compression", {})
                ),
                evaluation_config=self.get_or_create(
                    session, EvaluationConfigSql, exp.config.get("evaluation", {})
                ),
                network_config=self.get_or_create(
                    session, NetworkConfigSql, exp.config.get("network", {})
                ),
                results=results,
            )
            session.add(experiment)
            session.commit()
            print(f"Saved experiment {experiment.id} to database at {self.db_path}")
        except Exception as e:
            session.rollback()
            print(f"Error saving experiment: {e}")
        finally:
            session.close()

    def get_or_create(
        self, session: Session, model: Type[T], data: Dict[str, Any]
    ) -> T:
        """
        Get an existing database record or create a new one if it doesn't exist.

        Attempts to find a record matching the provided data. If no matching
        record is found, creates a new one with the given data.

        Args:
            session: The SQLAlchemy session to use for database operations.
            model: The SQLAlchemy model class to query or instantiate.
            data: Dictionary of attributes to filter by or use for creation.

        Returns:
            An instance of the model class, either existing or newly created.
        """
        instance = session.query(model).filter_by(**data).first()
        if not instance:
            instance = model(**data)
            session.add(instance)
            session.commit()
        return instance

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentSql]:
        """
        Retrieve an experiment from the database by its ID.

        Args:
            experiment_id: The unique identifier of the experiment to retrieve.

        Returns:
            The ExperimentSql instance if found, None otherwise.
        """
        session = self.Session()
        experiment = session.query(ExperimentSql).filter_by(id=experiment_id).first()
        session.close()
        return experiment

    def to_df(self) -> pd.DataFrame:
        """
        Convert all experiments and their related data to a flat pandas DataFrame.

        This method retrieves all experiments from the database and joins them with
        their related compression, calibration, evaluation, and results data to create
        a comprehensive flat DataFrame for analysis.

        Returns:
            A pandas DataFrame containing all experiment data with related configurations
            and results in a flat structure with column names prefixed by their source table.
        """
        session = self.Session()
        try:
            # Create a query that joins all related tables
            query = (
                session.query(
                    ExperimentSql,
                    CompressionConfigSql,
                    CalibrationConfigSql,
                    EvaluationConfigSql,
                    NetworkConfigSql,
                    ExperimentResultSql,
                )
                .join(
                    CompressionConfigSql,
                    ExperimentSql.compression_config_id == CompressionConfigSql.id,
                )
                .join(
                    CalibrationConfigSql,
                    ExperimentSql.calibration_config_id == CalibrationConfigSql.id,
                )
                .join(
                    EvaluationConfigSql,
                    ExperimentSql.evaluation_config_id == EvaluationConfigSql.id,
                )
                .join(
                    NetworkConfigSql,
                    ExperimentSql.network_config_id == NetworkConfigSql.id,
                )
                .join(
                    ExperimentResultSql,
                    ExperimentSql.results_id == ExperimentResultSql.id,
                )
            )

            # Execute the query
            results = query.all()

            if not results:
                return pd.DataFrame()

            # Convert to a flat dictionary structure
            flat_data = []
            for exp, comp, calib, eval_, network, res in results:
                data_dict = {
                    # Experiment columns
                    "id": exp.id,
                    "experiment_name": exp.experiment_name,
                    "experiment_description": exp.experiment_description,
                    "experiment_task": exp.experiment_task,
                    "timestamp": exp.timestamp,
                    # Compression columns
                    "compression_method": comp.method,
                    "compression_log10_lm": comp.log10_lm,
                    "compression_groupsize": comp.groupsize,
                    "compression_entropy_model": comp.entropy_model,
                    "compression_nbins": comp.nbins,
                    "compression_scan_order_major": comp.scan_order_major,
                    # Calibration columns
                    "calibration_dataset": calib.dataset,
                    "calibration_nbatches": calib.nbatches,
                    "calibration_batch_size": calib.batch_size,
                    # Evaluation columns
                    "evaluation_dataset": eval_.dataset,
                    "evaluation_nbatches": eval_.nbatches,
                    "evaluation_batch_size": eval_.batch_size,
                    "evaluation_entropy_model": eval_.entropy_model,
                    "evaluation_transpose": eval_.transpose,
                    # Evaluation columns
                    "network_name": network.name,
                    "network_train_dataset": network.train_dataset,
                    # Results columns
                    "performance": res.performance,
                    "bitrate": res.bitrate,
                }
                flat_data.append(data_dict)

            # Convert to DataFrame
            return pd.DataFrame(flat_data)
        finally:
            session.close()
