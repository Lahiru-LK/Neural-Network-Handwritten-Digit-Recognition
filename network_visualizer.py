import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Initialize accuracy & loss tracking
train_loss_history = []
train_acc_history = []

def visualize_network(screen, nn, input_size, hidden_size_1, hidden_size_2, output_size,
                      epoch, batch_idx, predicted_index=None,
                      train_loss=0, train_acc=0, test_loss=0, test_acc=0, max_epochs=10):
    """Neural network visualization with gradient-weighted connections & training accuracy graph."""

    global train_loss_history, train_acc_history

    screen.fill((0, 0, 0))  # Clear screen with black background

    # **Update Accuracy & Loss Graph Data**
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # **Define Positioning (Neural Network on Left, Stats & Graph on Right)**
    input_layer_x = 100
    hidden_layer_1_x = 300
    hidden_layer_2_x = 500
    output_layer_x = 700

    # **Define y-coordinates for neurons**
    input_layer_y = np.linspace(50, 550, input_size // 50)
    hidden_layer_1_y = np.linspace(50, 550, hidden_size_1 // 50)
    hidden_layer_2_y = np.linspace(50, 550, hidden_size_2 // 50)
    output_layer_y = np.linspace(50, 550, output_size)

    # **Draw Neurons**
    def draw_neurons(x, y_list, activations=None):
        for i, y in enumerate(y_list):
            if activations is not None:
                activation_value = max(0, min(1, activations[i]))
                color = (int(activation_value * 255), int(activation_value * 255), int(activation_value * 255))
            else:
                color = (255, 255, 255)
            pygame.draw.circle(screen, color, (x, int(y)), 8)

    # **Draw Connections (Gradient Coloring)**
    def draw_connections(x1, y1_list, x2, y2_list, weights):
        for i, y1 in enumerate(y1_list):
            for j, y2 in enumerate(y2_list):
                weight = weights[i, j] if i < weights.shape[0] and j < weights.shape[1] else 0
                alpha = int(min(255, max(50, 255 * abs(weight) * 10)))
                color = (0, alpha, 0) if weight > 0 else (alpha, 0, 0)
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)

    # **Draw Neural Network with Proper Connections**
    draw_neurons(input_layer_x, input_layer_y)
    draw_connections(input_layer_x, input_layer_y, hidden_layer_1_x, hidden_layer_1_y,
                     nn.weights_input_hidden1[:len(input_layer_y), :len(hidden_layer_1_y)])
    draw_neurons(hidden_layer_1_x, hidden_layer_1_y, nn.hidden1_output[0] if hasattr(nn, 'hidden1_output') else None)
    draw_connections(hidden_layer_1_x, hidden_layer_1_y, hidden_layer_2_x, hidden_layer_2_y,
                     nn.weights_hidden1_hidden2[:len(hidden_layer_1_y), :len(hidden_layer_2_y)])
    draw_neurons(hidden_layer_2_x, hidden_layer_2_y, nn.hidden2_output[0] if hasattr(nn, 'hidden2_output') else None)
    draw_connections(hidden_layer_2_x, hidden_layer_2_y, output_layer_x, output_layer_y,
                     nn.weights_hidden2_output[:len(hidden_layer_2_y), :len(output_layer_y)])
    draw_neurons(output_layer_x, output_layer_y, nn.output[0] if hasattr(nn, 'output') else None)

    # **Output Labels**
    font = pygame.font.Font(None, 36)
    for i, y in enumerate(output_layer_y):
        color = (255, 255, 0) if predicted_index == i else (255, 255, 255)
        text = font.render(str(i), True, color)
        screen.blit(text, (output_layer_x + 30, int(y) - 10))

    # **Right-Aligned Stats**
    font_big = pygame.font.Font(None, 42)
    font_small = pygame.font.Font(None, 30)

    stats_x = output_layer_x + 190
    stats_y_start = 50

    screen.blit(font_big.render(f"Epoch: {epoch + 1}/{max_epochs} | Batch: {batch_idx + 1}", True, (255, 255, 255)), (stats_x, stats_y_start))
    screen.blit(font_small.render(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%", True, (255, 255, 255)), (stats_x, stats_y_start + 40))

    # **Move Accuracy Graph to the Right Side**
    fig, ax = plt.subplots(figsize=(5, 4))  # 🔥 Increased Width & Height
    fig.patch.set_alpha(0)  # ✅ Transparent Background for Figure
    ax.set_facecolor((0, 0, 0, 0))  # ✅ Transparent Background for Axis

    # **Plot Accuracy Curve**
    ax.plot(train_acc_history, label="Train Accuracy", color='white', linewidth=0.5)  # ✅ White Line for Visibility

    # **Title & Labels in White for Readability**
    ax.set_title("Training Accuracy", fontsize=16, fontweight="bold", color='white')
    ax.set_xlabel("Iterations", fontsize=13, color='white')
    ax.set_ylabel("Accuracy (%)", fontsize=13, color='white')

    # **Improve Axis Visibility (White)**
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='both', colors='white')

    # **Legend with Dark Theme**
    ax.legend(fontsize=8, loc="upper left", facecolor="black", edgecolor="white", labelcolor="white")

    # **Convert Matplotlib Graph to Pygame Image**
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # **Get Image Data**
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()

    # **Create Pygame Surface & Blit to Screen**
    graph_surface = pygame.image.frombuffer(raw_data, size, "RGBA")

    screen.blit(graph_surface, (780, 140))  # ✅ Adjust Position on Right Side
    plt.close(fig)  # Close the figure after rendering

    pygame.display.flip()




