from torch import nn

def create_mlp(input_dim, hidden_layers, output_dim, dropout_rates=None, *,
               use_batchnorm=True,
               use_layernorm_at_output=False,
               init_method='xavier'):
    """
    Creates a flexible MLP model with optional dropout, batch norm, layer norm, and custom init.

    Parameters:
    - input_dim (int): Number of input features.
    - hidden_layers (list of int): List of hidden layer sizes.
    - output_dim (int): Output size.
    - dropout_rates (list of float): Dropout rate per hidden layer.
    - use_batchnorm (bool): Whether to apply BatchNorm after each Linear layer.
    - use_layernorm_at_output (bool): Whether to apply LayerNorm after the final output layer.
    - init_method (str): Initialization method. Currently supports 'xavier'.

    Returns:
    - nn.Sequential: The MLP model.
    """

    layers = []
    prev_dim = input_dim

    for i, hidden_dim in enumerate(hidden_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rates is not None:
            layers.append(nn.Dropout(p=dropout_rates[i]))
        prev_dim = hidden_dim

    # Final layer
    layers.append(nn.Linear(prev_dim, output_dim))
    if use_layernorm_at_output:
        layers.append(nn.LayerNorm(output_dim))

    model = nn.Sequential(*layers)

    # === Apply initialization ===
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model
