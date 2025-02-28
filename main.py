
import pygame
import numpy as np
from data_loader import load_data
from neural_network import NeuralNetwork
from utils import preprocess_digit
import traceback
from network_visualizer import visualize_network

# Constants
INPUT_SIZE = 784
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
OUTPUT_SIZE = 10
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def train_nn_with_visualization(nn, x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate):
    print("🚀 Training Started...")
    pygame.init()
    screen = pygame.display.set_mode((1300, 600))
    pygame.display.set_caption("Neural Network Training Visualization")

    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]

        epoch_loss = 0
        epoch_correct = 0

        for batch_idx, i in enumerate(range(0, x_train.shape[0], batch_size)):
            pygame.event.pump()

            x_batch, y_batch = x_train[i:i + batch_size], y_train[i:i + batch_size]
            outputs = nn.forward(x_batch)
            nn.backward(x_batch, y_batch, outputs, learning_rate)

            batch_loss = nn.cross_entropy_loss(y_batch, outputs)
            batch_predictions = np.argmax(outputs, axis=1)
            batch_true_labels = np.argmax(y_batch, axis=1)
            batch_correct = np.sum(batch_predictions == batch_true_labels)

            epoch_loss += batch_loss
            epoch_correct += batch_correct

            predicted_index = batch_predictions[0]

            visualize_network(
                screen, nn, INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE,
                epoch, batch_idx, predicted_index,
                epoch_loss, (epoch_correct / ((batch_idx + 1) * batch_size)) * 100, 0, 0, epochs
            )

        # Compute test accuracy
        test_outputs = nn.forward(x_test)
        test_loss = nn.cross_entropy_loss(y_test, test_outputs)
        test_predictions = np.argmax(test_outputs, axis=1)
        test_true_labels = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_predictions == test_true_labels) * 100

        print(f"🎯 Epoch {epoch + 1}/{epochs} -> Train Loss: {epoch_loss:.4f}, Train Accuracy: {(epoch_correct / x_train.shape[0]) * 100:.2f}%")
        print(f"📊 Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    pygame.quit()





# Function to get user input and predict digit
def get_user_input_and_predict(nn):
    pygame.init()
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("Draw a Digit")
    clock = pygame.time.Clock()

    drawing = False
    digit_image = np.zeros((28, 28))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                processed_image = preprocess_digit(digit_image)
                prediction = np.argmax(nn.forward(processed_image.reshape(1, -1)))

                print(f"🎯 Predicted Digit: {prediction}")  # Display the recognized digit

                digit_image.fill(0)  # Clear the screen for the next digit

        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if 0 <= mouse_x < 280 and 0 <= mouse_y < 280:
                grid_x = mouse_x // 10
                grid_y = mouse_y // 10
                digit_image[grid_y, grid_x] = 255
                pygame.draw.circle(screen, (255, 255, 255), (mouse_x, mouse_y), 10)

        screen.fill((0, 0, 0))
        for y in range(28):
            for x in range(28):
                if digit_image[y, x] > 0:
                    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(x * 10, y * 10, 10, 10))

        pygame.display.flip()
        clock.tick(30)


# Main function
def main():
    try:
        print("🔍 Initializing Neural Network...")
        x_train, x_test, y_train, y_test = load_data()
        nn = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
        print("✅ Neural Network Initialized!")

        train_nn_with_visualization(nn, x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

        print("✅ Training complete! 🎉 You can now draw a digit on the screen.")
        get_user_input_and_predict(nn)

    except Exception as e:
        print("❌ ERROR OCCURRED:", str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
