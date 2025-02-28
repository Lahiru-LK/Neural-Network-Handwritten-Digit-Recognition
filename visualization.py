import pygame
import numpy as np

def visualize_network(screen, nn, input_size, hidden_size_1, hidden_size_2, output_size, epoch, batch_idx, predicted_index=None):
    """Draws the neural network layers and connections during training."""

    screen.fill((0, 0, 0))  # Clear the screen

    # Define positions for layers
    input_layer_x = 100
    hidden_layer_1_x = 300
    hidden_layer_2_x = 500
    output_layer_x = 700

    # Define y-coordinates for neurons in each layer
    input_layer_y = np.linspace(50, 550, input_size // 50)
    hidden_layer_1_y = np.linspace(50, 550, hidden_size_1 // 50)
    hidden_layer_2_y = np.linspace(50, 550, hidden_size_2 // 50)
    output_layer_y = np.linspace(50, 550, output_size)

    # Function to draw neurons
    def draw_neurons(x, y_list, activations=None):
        for i, y in enumerate(y_list):
            if activations is not None:
                activation_value = max(0, min(1, activations[i]))  # Normalize activations
                color = (int(activation_value * 255), int(activation_value * 255), int(activation_value * 255))  # Grayscale
            else:
                color = (255, 255, 255)  # Default to white
            pygame.draw.circle(screen, color, (x, int(y)), 5)

    # Function to draw connections
    def draw_connections(x1, y1_list, x2, y2_list, weights):
        for i, y1 in enumerate(y1_list):
            for j, y2 in enumerate(y2_list):
                weight = weights[i, j] if i < weights.shape[0] and j < weights.shape[1] else 0
                color_intensity = int(min(255, max(50, 255 * abs(weight) * 10)))
                color = (0, color_intensity, 0) if weight > 0 else (color_intensity, 0, 0)
                pygame.draw.line(screen, color, (x1, y1), (x2, y2))

    # Draw each layer and its connections
    draw_neurons(input_layer_x, input_layer_y)
    draw_connections(input_layer_x, input_layer_y, hidden_layer_1_x, hidden_layer_1_y, nn.weights_input_hidden1[:len(input_layer_y), :len(hidden_layer_1_y)])
    draw_neurons(hidden_layer_1_x, hidden_layer_1_y, nn.hidden1_output[0] if hasattr(nn, 'hidden1_output') else None)
    draw_connections(hidden_layer_1_x, hidden_layer_1_y, hidden_layer_2_x, hidden_layer_2_y, nn.weights_hidden1_hidden2[:len(hidden_layer_1_y), :len(hidden_layer_2_y)])
    draw_neurons(hidden_layer_2_x, hidden_layer_2_y, nn.hidden2_output[0] if hasattr(nn, 'hidden2_output') else None)
    draw_connections(hidden_layer_2_x, hidden_layer_2_y, output_layer_x, output_layer_y, nn.weights_hidden2_output[:len(hidden_layer_2_y), :len(output_layer_y)])
    draw_neurons(output_layer_x, output_layer_y, nn.output[0] if hasattr(nn, 'output') else None)

    # Draw numbers on the output layer
    font = pygame.font.Font(None, 36)
    for i, y in enumerate(output_layer_y):
        color = (255, 255, 0) if predicted_index == i else (255, 255, 255)  # Highlight predicted neuron
        text = font.render(str(i), True, color)
        screen.blit(text, (output_layer_x + 20, int(y) - 10))

    # Display epoch and batch information
    info_font = pygame.font.Font(None, 36)
    text = info_font.render(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()  # Update the display
