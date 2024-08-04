import torch
import torch.nn.functional as F


def var_detection(
        logits: torch.Tensor,
        alpha: float
):
    with torch.no_grad():
        next_token_logits = logits[:, -1, :][0]
        variance = torch.var(next_token_logits, dim=0)
        return variance < alpha




def JSD_detection(
        hidden_states: tuple,
        mature_layer: int,
        alpha: float
):
    output_list = list()

    for i, layer_output in enumerate(hidden_states):
        output_list.append(layer_output)

    candidate_premature_layers = len(output_list) // 2
    assert mature_layer >= len(output_list) / 2 and mature_layer < len(output_list) - 1

    # for i in range(len(output_list) // 2):

    # 1. Stacking all premature_layers into a new dimension
    stacked_premature_layers = torch.stack([output_list[i] for i in range(candidate_premature_layers)],
                                           dim=0)
    # 2. Calculate the softmax values for mature_layer and all premature_layers
    softmax_mature_layer = F.softmax(output_list[mature_layer],
                                     dim=-1)  # shape: (batch_size, num_features)
    softmax_premature_layers = F.softmax(stacked_premature_layers,
                                         dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

    # 3. Calculate M, the average distribution
    M = 0.5 * (softmax_mature_layer[None, :,
               :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(output_list[mature_layer],
                                             dim=-1)  # shape: (batch_size, num_features)
    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers,
                                                 dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(
        -1)  # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(
        -1)  # shape: (num_premature_layers, batch_size)
    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

    # 6. Reduce the batchmean
    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers)

    # 7. Get max JSD and the use to detection
    JSD = js_divs.max()

    return JSD < alpha