
import tensorflow
# module qui initialise le reseau de neurone
from tensorflow.keras.models import Sequential
# permet la creation des couches de neurones ( entree - cachee - sortie )
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten
# ajout de Dropout pour limiter le sur-apprentissage
from tensorflow.keras.layers import Dropout
import json
import random

optimizers = ['sgd', 'rmsprop', 'adagrad',
              'adadelta', 'adam', 'adamax', 'nadam']
activations = ['relu', 'softmax', 'elu', 'selu', 'softplus', 'softsign',
               'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']


def model_generator(list_elem, exit_number):
    model = Sequential()

    # Traitement si image
    if(list_elem['is_image'] == True):
        nb_filter = 0
        for i in range(len(list_elem['pooling'])):
            item = list_elem['pooling'][i]

            if(i == 0):
                model.add(Conv2D(2**(4+i), kernel_size=(3, 3), input_shape=(64,
                                                                            64, 3), activation=activations[item['activation']]))
            else:
                if(item["increase_filter"] and i < 4):
                    nb_filter += 1
                model.add(Conv2D(2**(4+i), kernel_size=(3, 3),
                                 activation=activations[item['activation']], padding="same"))

            model.add(BatchNormalization())
            model.add(MaxPooling2D())
            if(item['with_dropout'] == True):
                model.add(Dropout(item['dropout']))
        model.add(Flatten())

    for i in range(len(list_elem['dense'])):
        dense = list_elem['dense'][i]

        model.add(Dense(
            units=dense['units'], activation=activations[dense['activation']], kernel_initializer="uniform"))
        if(dense['with_dropout'] == True):
            model.add(Dropout(dense['dropout']))
    model.add(Dense(units=exit_number,
                    activation=activations[list_elem['last_activation']], kernel_initializer="uniform"))

    loss = None
    if(list_elem['is_image']):
        loss = "categorical_crossentropy"
    else:
        loss = "mse"
    model.compile(
        # optimizer=optimizers[list_elem['optimizer']], loss=loss, metrics=["accuracy"])
        optimizer='adam', loss=loss, metrics=["accuracy"])
    return model


def get_new_random_dense(max_unit, nb_output):
    return {
        "units": get_new_units(max_unit, nb_output),
        "activation": get_new_activation(),
        "dropout": get_new_dropout(),
        "with_dropout": should_add_dropout()
    }


def get_new_random_pooling():
    return {
        "activation": get_new_activation(),
        "with_dropout": should_add_dropout(),
        "dropout": get_new_dropout(),
        "increase_filter": should_increase_filter()
    }


def should_add_dropout():
    val = round(random.uniform(0, 1), 1)
    return val >= 0.7


def should_increase_filter():
    return round(random.uniform(0, 1), 1) <= 0.2


def get_new_dropout():
    return round(random.uniform(0.1, 0.5), 2)


def get_new_units(max_unit, nb_output):
    return random.randint(max(round(max_unit*0.7), nb_output), max_unit)


def get_new_activation():
    return random.randint(0, len(activations)-1)


def get_new_optimizer():
    return random.randint(0, len(optimizers)-1)


def get_nb_dense():
    return random.randint(1, 3)


def get_nb_pooling():
    return random.randint(1, 4)


def get_nb_epochs():
    return 5


def get_batch_size():
    return random.randint(1, 100)


def choose_parameter(first_choice, second_choice, new_choice):
    value = random.uniform(0, 1)
    if value < 0.45:
        return first_choice
    if value < 0.90:
        return second_choice
    return new_choice
