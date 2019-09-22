import numpy
import scipy.special as special


class neuralNetwork:

    # Инициализируем нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Задаем количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Коэффициент обучения
        self.lr = learningrate

        # Матрица весовых коэффициентов связей wih(между входным и скрытым слоями) и who(между скрытым и
        # выходным слоями).
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes - 0.5))
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        # self.who = (numpy.random.rand(self.onodes, self.hnodes - 0.5))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # Использование сигмоиды, как активационной функции
        self.activation_function = lambda x: special.expit(x)
        pass

    # Тренировка нейронной сети
    def train(self, input_list, target_list):
        # Преоброзавоть список входных значений в двумерный массив
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # Рассчитать входные сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Рассчитать входные сигналы для последнего слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Рассчитать исходящие сигналы для последнего слоя
        final_outputs = self.activation_function(final_inputs)

        # Ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # Ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей и рекомбинированные на
        # на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # Обновить весовые коэффициенты связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # Опрос нейронной сети
    def query(self, input_list):
        # Преобразовать список входных значений в двумерный массив
        inputs = numpy.array(input_list, ndmin=2).T

        # Рассчитать входящие сигналы для скрытого слоя
        hiden_inputs = numpy.dot(self.wih, inputs)
        # Рассчитать выходящие сигналы для скрытого слоя
        hiden_outputs = self.activation_function(hiden_inputs)

        # Рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hiden_outputs)
        # Рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
