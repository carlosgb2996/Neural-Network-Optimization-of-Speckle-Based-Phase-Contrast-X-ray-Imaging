import torch
import torch.nn.functional as F

# Calcula el error de punto final (EPE, End-Point Error) entre dos mapas de flujo.
# Este error mide la distancia euclidiana entre los vectores correspondientes en los mapas de flujo predicho y objetivo.
def EPE(input_flow, target_flow, sparse=False, mean=True):
    """
    Calcula el Error de Punto Final (EPE) entre el flujo predicho y el flujo objetivo.

    Args:
        input_flow (torch.Tensor): Flujo predicho con forma [batch, 2, H, W].
        target_flow (torch.Tensor): Flujo objetivo con la misma forma que input_flow.
        sparse (bool): Si el flujo objetivo es disperso (contiene regiones inválidas marcadas como 0).
        mean (bool): Si se devuelve la media o la suma del error EPE.

    Returns:
        torch.Tensor: Valor escalar del error EPE.
    """
    # Calcula la magnitud de la diferencia entre los flujos en cada posición.
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)  # Norma L2 a lo largo de la dimensión del flujo (dimensión 1).
    batch_size = EPE_map.size(0)  # Número de muestras en el lote.

    if sparse:
        # Identifica regiones inválidas en el flujo objetivo (ambos componentes son exactamente 0).
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        # Filtra las regiones inválidas del mapa EPE.
        EPE_map = EPE_map[~mask]

    if mean:
        return EPE_map.mean()  # Devuelve la media del error EPE.
    else:
        return EPE_map.sum() / batch_size  # Devuelve la suma normalizada por el tamaño del lote.

# Aplica un "max pooling" adaptativo en un flujo disperso.
def sparse_max_pool(input, size):
    """
    Realiza un max pooling adaptativo en un flujo disperso, manejando valores positivos y negativos por separado.

    Args:
        input (torch.Tensor): Flujo de entrada con forma [batch, channels, H, W].
        size (tuple): Dimensiones de salida deseadas (alto, ancho).

    Returns:
        torch.Tensor: Flujo de salida después del max pooling.
    """
    # Máscaras para valores positivos y negativos.
    positive = (input > 0).float()
    negative = (input < 0).float()

    # Aplica max pooling por separado a los valores positivos y negativos.
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output

# Calcula el error EPE multiescala, combinando errores en diferentes resoluciones.
def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    """
    Calcula el EPE multiescala combinando errores a varias resoluciones del flujo predicho.

    Args:
        network_output (list or tuple): Salidas del modelo en múltiples escalas.
        target_flow (torch.Tensor): Flujo objetivo con forma [batch, 2, H, W].
        weights (list): Pesos para cada escala en el cálculo del error.
        sparse (bool): Si el flujo objetivo es disperso.

    Returns:
        torch.Tensor: Pérdida total ponderada en todas las escalas.
    """
    def one_scale(output, target, sparse):
        """
        Calcula el EPE para una sola escala, ajustando el flujo objetivo al tamaño de la salida.

        Args:
            output (torch.Tensor): Flujo predicho en la escala actual.
            target (torch.Tensor): Flujo objetivo original.
            sparse (bool): Si el flujo objetivo es disperso.

        Returns:
            torch.Tensor: Error EPE para esta escala.
        """
        b, _, h, w = output.size()  # Dimensiones de la salida (batch, channels, height, width).

        if sparse:
            # Ajusta el flujo objetivo a la escala actual usando max pooling.
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            # Ajusta el flujo objetivo usando interpolación por área.
            target_scaled = F.interpolate(target, (h, w), mode='area')

        # Calcula el EPE para esta escala sin promediar.
        return EPE(output, target_scaled, sparse, mean=False)

    # Si la salida del modelo no es una lista o tupla, la convierte en una lista.
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]

    # Define los pesos predeterminados si no se especifican.
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # Pesos según el artículo original.

    # Asegura que el número de pesos coincida con el número de escalas.
    assert(len(weights) == len(network_output))

    # Calcula la pérdida ponderada en todas las escalas.
    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)

    return loss

# Calcula el error EPE para la salida del modelo interpolada a la resolución original.
def realEPE(output, target, sparse=False):
    """
    Calcula el EPE interpolando la salida del modelo a la resolución del flujo objetivo.

    Args:
        output (torch.Tensor): Flujo predicho con forma [batch, 2, H_out, W_out].
        target (torch.Tensor): Flujo objetivo con forma [batch, 2, H, W].
        sparse (bool): Si el flujo objetivo es disperso.

    Returns:
        torch.Tensor: Error EPE interpolado.
    """
    b, _, h, w = target.size()  # Dimensiones del flujo objetivo.

    # Interpola la salida del modelo a la resolución del flujo objetivo.
    upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)

    # Calcula el EPE entre la salida interpolada y el flujo objetivo.
    return EPE(upsampled_output, target, sparse, mean=True)