# fmt: off
METHOD_LINESTYLE   = "-"
BASELINE_LINESTYLE = "dashdot"

OPTQ_SYMBOL      = "x"
OPTQ_RD_SYMBOL   = "."
OPTQ_RD_R_SYMBOL = "^"
RTN_SYMBOL = "o"
nnc_SYMBOL = ">"

COLORS = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]
c0 = COLORS[0]
c1 = COLORS[1]
c2 = COLORS[2]
c3 = COLORS[3]
c4 = COLORS[4]
c5 = COLORS[5]
c6 = COLORS[6]
c7 = COLORS[7]

BPW_LABEL           = "Bits Per Weight"
SPARSITY_LABEL      = "Sparsity"
INV_SPARSITY_LABEL  = "Non-Zero Weights"
DEFAULT_MARKER      = "--."
NO_MARKER_LINEWIDTH = 1.1

AXIS_LABELS_CV      = [BPW_LABEL, "Top-1 Accuracy"]
AXIS_LABELS_NLP     = [BPW_LABEL, "Perplexity"]
DEFAULT_LINESTYLE   = "-"
MARKEVERY           = 0.02
DASH_STYLE          = (3, 1)

COLORS_BARS = {
    "Hessian": "#00678a",
    "GPTQ Quant.": "#984464",
    "OPTQ-RD Quant.": "#56641a",
    "DeepCABAC Encode": "#5eccab",
    "DeepCABAC Decode": "#e6a176",
}

PLOT_PARAMS_METHODS = {
    "optq": {
        "linestyle": METHOD_LINESTYLE,
        "dashes": [],
        "marker": OPTQ_SYMBOL,
        "color": c0,
        "markersize": 12,
    },
    "optq-rd": {
        "linestyle": METHOD_LINESTYLE,
        "marker": OPTQ_RD_SYMBOL,
        "color": c1,
    },
    "optq-rd-r": {
        "linestyle": METHOD_LINESTYLE,
        "marker": OPTQ_RD_R_SYMBOL,
        "color": c2,
        "markersize": 4,
    },  
    "rtn": {
        "linestyle": BASELINE_LINESTYLE,
        "marker": RTN_SYMBOL,
        "color": c3,
        "markersize": 3,
    },
    "nnc": {
        "linestyle": BASELINE_LINESTYLE,
        "marker": nnc_SYMBOL,
        "color": c4,
        "markersize": 3,
    },
    "base_performance": {
        "linestyle": "--",
        "color": "grey",
        "alpha": 0.4
    },
}

PLOT_LABELS = {
    "optq": r"CERWU-$\lambda\!=\!0$",
    "optq-rd": r"CERWU-$\gamma\!=\!0$",
    "optq-rd-r": "CERWU",
    "nnc": "NNCodec",
    "rtn": "RTN+EC"
}

NETWORK_LABELS = {
    "resnet18_cifar10": "ResNet18 (CIFAR10)",
    "resnet34_cifar10": "ResNet34 (CIFAR10)",
    "resnet50_cifar10": "ResNet50 (CIFAR10)",

    "resnet18_imagenet": "ResNet18 (ImageNet)",
    "resnet34_imagenet": "ResNet34 (ImageNet)",
    "resnet50_imagenet": "ResNet50 (ImageNet)",
    "resnet101_imagenet": "ResNet101 (ImageNet)",
    "resnet152_imagenet": "ResNet152 (ImageNet)",

    "vgg16": "VGG16",
    "pythia70m": "Pythia-70M",
    "mobilenetv3_small": "MobileNetv3 (Small)",
    "mobilenetv3_large": "MobileNetv3 (Large)",
}


BASELINE_PERF = {
 'resnet18_cifar10': 0.9498,
 'resnet34_cifar10': 0.954,
 'resnet50_cifar10': 0.9465,

 'resnet18_imagenet': 0.7006875,#
 'resnet34_imagenet': 0.73025,#
 'resnet50_imagenet': 0.7625,#

 'resnet101_imagenet': 0.777,
 'resnet152_imagenet': 0.78925,
 'vgg16': 0.716375,#
 'pythia70m': 48.739505767822266 ,#
 'mobilenetv3_small': 0.67475,#
 'mobilenetv3_large': 0.75225,#
}

RUN_TIMES_VGG16 = {
    "hessian": 1526,
    "rtn": 3,
    "optq-rd-encode_16": 975,
    "optq-rd-encode_32": 1136,
    "optq-encode_16": 1088,
    "optq-encode_32": 1250,
    "cerwu-encode_16": 1672.25,
    "cerwu-encode_32": 2047,
    "nnc": 14,
    "decode": 3
}

NQUANTISABLE_PARAMS = {'resnet18_cifar10': 11159232,
 'resnet34_cifar10': 21259968,
 'resnet50_cifar10': 23447232,
 'resnet18_imagenet': 11166912,
 'resnet34_imagenet': 21267648,
 'resnet50_imagenet': 23454912,
 'resnet101_imagenet': 42394816,
 'resnet152_imagenet': 57992384,
 'mobilenetv3_large': 5451272,
 'mobilenetv3_small': 2525832,
 'vgg16': 138344128,
 'convnext_tiny': 28524000,
 'pythia-70m': 18874368}

# check job 1510846_xxxx
RUN_TIMES_RESNET = {
    "resnet18_imagenet": {
        "bins": [4,8,16,32,64,128,256],
        "times": [36, 41, 52, 78, 142, 264, 494]
    },
    "resnet34_imagenet": {
        "bins": [4,8,16,32,64,128,256],
        "times": [57, 68, 90, 143, 258, 480, 940] # missing 13
    },
    "resnet50_imagenet": {
        "bins": [4,8,16,32,64,128,256],
        "times": [48, 57, 98, 164, 263, 512, 1022] # missing 20
    },
    "resnet101_imagenet": {
        "bins": [4,8,16,32,64,128,256],
        "times": [83, 100, 150, 255, 467, 920, 1841] # 27
    },
    "resnet152_imagenet": {
        "bins": [4,8,16,32,64,128,256],
        "times": [117, 138, 214, 339, 637, 1364, 2507] # 33 missing
    },
}

HESSIAN_CALC_TIMES = {
    "resnet18_imagenet": 659,
    "resnet34_imagenet": 683,
    "resnet50_imagenet": 768,
    "resnet101_imagenet": 1380,
    "resnet152_imagenet": 1622,
}

ENCODE_TIMES_RESNET_NET_NNC = {
    "resnet18_imagenet": 1.06,
    "resnet34_imagenet": 2.2,
    "resnet50_imagenet": 3.4,
    "resnet101_imagenet": 6.8,
    "resnet152_imagenet": 11.8,
}

# check 1510937_0
DECODE_TIMES_RESNET = {
    "resnet18_imagenet": 0.23,
    "resnet34_imagenet": 0.5,
    "resnet50_imagenet": 0.5, 
    "resnet101_imagenet": 1,
    "resnet152_imagenet": 1.2
}

OVERHEADS = {'resnet18_cifar10': 2.8675808514421065e-05,
 'resnet34_cifar10': 2.7093173423403083e-05,
 'resnet50_cifar10': 3.616631592164056e-05,
 'resnet18_imagenet': 2.8656086839405558e-05,
 'resnet34_imagenet': 2.7083389757061994e-05,
 'resnet50_imagenet': 3.6154473740937504e-05,
 'resnet101_imagenet': 3.92500818968055e-05,
 'resnet152_imagenet': 4.2764236076240634e-05,
 'mobilenetv3_large': 0.0148662550685418,
 'mobilenetv3_small': 0.01583003145102287,
 'vgg16': 1.8504580114885686e-06,
 'convnext_tiny': 0.003738606086102931,
 'pythia-70m': 2.0345052083333332e-05}
