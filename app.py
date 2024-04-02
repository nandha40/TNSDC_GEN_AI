import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def initialise(a, b):
    epsilon = 0.15
    c = np.random.rand(a, b + 1) * (2 * epsilon) - epsilon 
    return c

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1)
    z2 = np.dot(X, Theta1.transpose())
    a2 = 1 / (1 + np.exp(-z2)) 
    one_matrix = np.ones((m, 1))
    a2 = np.append(one_matrix, a2, axis=1) 
    z3 = np.dot(a2, Theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3)) 
    p = (np.argmax(a3, axis=1))
    return p

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                        (num_labels, hidden_layer_size + 1))

    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1) 
    a1 = X
    z2 = np.dot(X, Theta1.transpose())
    a2 = 1 / (1 + np.exp(-z2))
    one_matrix = np.ones((m, 1))
    a2 = np.append(one_matrix, a2, axis=1)
    z3 = np.dot(a2, Theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3))

    y_vect = np.zeros((m, 10))
    for i in range(m):
        y_vect[i, int(y[i])] = 1

    J = (1 / m) * (np.sum(np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)))) + (lamb / (2 * m)) * (
                sum(sum(pow(Theta1[:, 1:], 2))) + sum(sum(pow(Theta2[:, 1:], 2))))

    Delta3 = a3 - y_vect
    Delta2 = np.dot(Delta3, Theta2) * a2 * (1 - a2)
    Delta2 = Delta2[:, 1:]

    Theta1[:, 0] = 0
    Theta1_grad = (1 / m) * np.dot(Delta2.transpose(), a1) + (lamb / m) * Theta1
    Theta2[:, 0] = 0
    Theta2_grad = (1 / m) * np.dot(Delta3.transpose(), a2) + (lamb / m) * Theta2
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return J, grad

if __name__ == "__main__":
    data = loadmat('mnist-original.mat')

    X = data['data']
    X = X.transpose()

    X = X / 255

    y = data['label']
    y = y.flatten()

    X_train = X[:60000, :]
    y_train = y[:60000]

    X_test = X[60000:, :]
    y_test = y[60000:]

    m = X.shape[0]
    input_layer_size = 784 
    hidden_layer_size = 100
    num_labels = 10 

    initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
    initial_Theta2 = initialise(num_labels, hidden_layer_size)

    initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
    maxiter = 50  
    lambda_reg = 0.1 
    myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

    accuracies = []  # Store accuracies for each epoch

    for epoch in range(1, maxiter + 1):
        results = minimize(neural_network, x0=initial_nn_params, args=myargs, 
                options={'disp': True, 'maxiter': epoch}, method="L-BFGS-B", jac=True)

        nn_params = results["x"] 

        Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
                                hidden_layer_size, input_layer_size + 1)) 
        Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                            (num_labels, hidden_layer_size + 1)) 

        pred = predict(Theta1, Theta2, X_test)
        accuracy = np.mean(pred == y_test) * 100
        accuracies.append(accuracy)

        print(f'Epoch {epoch}, Test Set Accuracy: {accuracy:.2f}%')

    plt.plot(range(1, maxiter + 1), accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.show()
    np.savetxt('file1.txt', Theta1, delimiter=' ')
    np.savetxt('file2.txt', Theta2, delimiter=' ')

    # Plot some examples from the test set with predicted and actual labels
    num_examples = 10
    indices = np.random.choice(len(X_test), num_examples, replace=False)
    predicted_labels = predict(Theta1, Theta2, X_test[indices])
    actual_labels = y_test[indices]

    plt.figure(figsize=(12, 8))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {predicted_labels[i]}, Actual: {actual_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predict(Theta1, Theta2, X_test))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
