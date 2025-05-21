from dataclasses import dataclass
import numpy as np
import torch
import math
from typing import Optional, Tuple, Iterator
from data_utils.arrays import strided_index_pairs
from data_utils.experiments import TimerLoop, Timer
from nn_compression.networks import LayerWiseHessian


class RegularizedCholeskyApproximator:
    @dataclass
    class ApproximationResult:
        """Results dataclass for the regularized Cholesky approximation."""

        inverse_regularized_hessian: torch.Tensor
        cholesky_of_inverse: torch.Tensor
        gamma: float
        delta: float

        def float(self):
            self.inverse_regularized_hessian = self.inverse_regularized_hessian.float()
            self.cholesky_of_inverse = self.cholesky_of_inverse.float()
            if isinstance(self.gamma, torch.Tensor):
                self.gamma = float(self.gamma.item())
            if isinstance(self.delta, torch.Tensor):
                self.delta = float(self.delta.item())
            return self

    def __init__(
        self,
        ws: torch.Tensor,
        H_init: torch.Tensor,
        lm: float,
        deltas: Optional[torch.Tensor] = None,
        n_iter: int = 10,
        groupsize: int = 1,
        debug: bool = True,
        reverse_order: bool = False,
        return_float: bool = True,
        verbose: bool = False,
        no_approx: bool = False,
        device: str = "cpu",
    ):
        """Approximates successive cholesky decompositions of inverses of slightly changing matrices:
            H_1 = H_init + lm * g1 * I
            H_2 = H_init + lm * g2 * I
            H_3 = H_init + lm * g3 * I
            ...
        ws: _Scaled_ weight parameter (must be approximately integer-valued)
        H_init: initial Hessian matrix (_unscaled_)
        deltas: groupwise values used to scale the original weight matrix W
        debug: If set to True, reports complete history of approximations
        """
        self.ws = ws.double()
        self.device = device
        self.groupsize = groupsize
        self.gammas = self.get_gammas(self.ws, groupsize)
        self.deltas = deltas.double() if deltas is not None else deltas
        self.verbose = verbose
        self.no_approx = no_approx

        # left in as we might want to later quantize rows in different orders
        self.order = np.arange(0, self.gammas.numel(), 1)

        self.gammas_ordered = self.gammas[self.order]
        self.i = 0

        H_init = H_init.double().to(self.device)

        self.eigvals, self.eigvecs = LayerWiseHessian.safe_pd_linalg(torch.linalg.eigh)(
            H_init
        )

        self.eigvals = self.eigvals.to("cpu")
        self.eigvecs = self.eigvecs.to("cpu")
        H_init = H_init.to("cpu")

        self.lm = lm
        self.n_iter = n_iter
        self.debug = debug
        self.return_float = return_float

        # Store the index pairs for later iteration
        self.index_pairs = list(strided_index_pairs(ws.shape[0], groupsize))

        #
        self._consumed = False  # indicates that the iterator has ran already

    def _Hinv_reg_from_eigvals(self, delta: float, gamma: float) -> torch.Tensor:
        # implicitly scaling H with delta here
        eigvals_regularized = 1 / (self.eigvals / delta**2 + self.lm * gamma)
        Hinv_reg = self.eigvecs @ torch.diag(eigvals_regularized) @ self.eigvecs.T
        return Hinv_reg

    def _cholesky_by_hand(self, Hinv_reg: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(Hinv_reg, upper=True)

    @staticmethod
    def get_gammas(w, groupsize):
        gammas = []
        for l, u in strided_index_pairs(w.shape[0], groupsize):
            w_slice = w[l:u]
            gammas.append(1 / (math.log(2) * torch.var(w_slice).item()))
        return torch.tensor(gammas)

    @property
    def current_delta(self) -> float:
        return self.deltas[self.order[self.i]].item() if self.deltas is not None else 1

    @property
    def last_delta(self) -> float:
        if self.deltas is None:
            return 1
        if self.i == 0:
            return self.current_delta
        else:
            return self.deltas[self.order[self.i - 1]].item()

    def get_next(self) -> ApproximationResult:
        if self.i >= self.gammas.numel():
            raise ValueError("Called get_next too often")

        delta = self.current_delta
        gamma = self.gammas_ordered[self.i].item()

        # implicitly scaling H with delta here
        Hinv_reg = self._Hinv_reg_from_eigvals(delta, gamma)
        chol = self._cholesky_by_hand(Hinv_reg)

        self.last_chol = chol
        current_gamma = self.gammas_ordered[self.i]
        self.i += 1

        approx = self.ApproximationResult(
            inverse_regularized_hessian=Hinv_reg,
            cholesky_of_inverse=chol,
            gamma=current_gamma.item(),
            delta=delta,
        )
        if self.return_float:
            approx = approx.float()
        return approx

    def __iter__(self) -> Iterator[Tuple[Tuple[int, int], ApproximationResult]]:
        """Make the class iterable, yielding (index_pair, results) tuples."""
        self.i = 0
        self._consumed = False
        return self

    def __next__(self) -> Tuple[Tuple[int, int], ApproximationResult]:
        """Return the next iteration result."""
        if self._consumed:
            raise ValueError(
                "Approximator object has already been consumed via iteration. Please re-initialize the object."
            )
        if self.i >= self.gammas.numel():
            self._consumed = True
            raise StopIteration

        # Get the original index pair using the ordering
        original_idx = self.order[self.i].item()
        index_pair = self.index_pairs[original_idx]  # type: ignore

        # Get the approximation result
        result = self.get_next()

        return index_pair, result
