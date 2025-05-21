import copy
import torch.nn as nn
import torch


def _recursively_find_named_children(top_name: str, net: nn.Module):
    all_children = []
    if len(list(net.children())) == 0:
        return [(top_name, net)]
    for name, child in net.named_children():
        children_rec = _recursively_find_named_children(name, child)
        if top_name != "":
            children_rec = [
                (top_name + "." + child_name, child)
                for child_name, child in children_rec
            ]
        all_children.extend(children_rec)
    return all_children


def recursively_find_named_children(net: nn.Module):
    """Recursively find all named children of a network and returns an iterator (string, nn.Module)."""
    return _recursively_find_named_children("", net)


def map_net(net: nn.Module, f) -> nn.Module:
    """Apply a function to all layers of a network, statically. f accepts a module and modifies it in-place.

    The whole network is not modified in place but returned."""
    net = copy.deepcopy(net)
    for name, child in recursively_find_named_children(net):
        f(child)
    return net


def map_net_forward(
    net: nn.Module,
    x: dict | torch.Tensor,
    f,
    require_name: bool = False,
    inplace: bool = False,
):
    """Apply a function to all layers of a network, while running an input x
    through the network. f has the same function signature as a hook:

    (module, input, output) -> None or modified output

    If require_name is set to true, the function f must have a name argument in the first place:

    (name, module, input, output) -> None or modified output
    """
    if inplace:
        net_mapped = net
    else:
        net_mapped = copy.deepcopy(net)

    handles = []
    named_children = recursively_find_named_children(net_mapped)
    for name, child in named_children:
        if require_name:
            f_curry = lambda name: lambda m, i, o: f(
                name, m, i, o
            )  # force evaluation of name
            handles.append(child.register_forward_hook(f_curry(name)))
        else:
            handles.append(child.register_forward_hook(f))

    if isinstance(x, torch.Tensor):
        net_mapped(x)
    else:
        net_mapped(**x)
    for handle in handles:
        handle.remove()
    return net_mapped
