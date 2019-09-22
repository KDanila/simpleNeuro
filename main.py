import neuro.neuralNetwork as nn
import numpy
import matplotlib.pyplot as plot

input_nodes = 784
hidden_nodes = 100000
output_nodes = 10

learning_rate = 0.3

n = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Тренировка нейронной сети

# Преобразовываем все данные в тренировочный набор данных
for record in training_data_list:
    # Получаем значения, использую символ ','
    all_values = record.split(',')
    # Масштабируем и смещаем  входные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

# Загружаем тестовые данные
test_data_file = open("dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',')
print("ЦИФРА = ", all_values[0])

image_array = numpy.asfarray(all_values[1:]).reshape([28, 28])
plot.imshow(image_array, cmap='Greys', interpolation='None')
plot.show()

disp = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
print(disp)
# print(scaled_input)
# plot.imshow(image_array, cmap='Greys', interpolation='None')
# plot.show()
# rand3x3 = numpy.random.rand(input_nodes, hidden_nodes)
# rand3x3 = (numpy.random.normal(0.0, pow(input_nodes, -0.5), (input_nodes, hidden_nodes)))
# print(rand3x3)
# plot.imshow(rand3x3);
# plot.show()
