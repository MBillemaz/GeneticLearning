import model_generator
import random


class Genetic_algo:
    def __init__(self, nb_output, is_image):
        self.nb_output = nb_output
        self.is_image = is_image

    def get_first_pop(self, nb_pop):
        config = []

        for i in range(nb_pop):
            max_unit = 200
            item = {
                "is_image": self.is_image,
                "pooling": [],
                "dense": [],
                "optimizer": model_generator.get_new_optimizer(),
                "last_activation": model_generator.get_new_activation(),
                # "epochs": random.randint(10, 1000),
                "epochs": model_generator.get_nb_epochs(),
                "batch_size": model_generator.get_batch_size()
            }

            if(self.is_image):
                for j in range(model_generator.get_nb_pooling()):
                    pool = model_generator.get_new_random_pooling()
                    item["pooling"].append(pool)

            for j in range(model_generator.get_nb_dense()):
                dense = model_generator.get_new_random_dense(
                    max_unit, self.nb_output)
                max_unit = dense['units']
                item["dense"].append(dense)
            config.append(item)

        return config

    def new_generation(self, old_pop):
        # For creating a new generation, we use the 3 best models and 2 random
        used_pop = old_pop[:3]
        random_pop = random.sample(range(3, 10), 2)
        used_pop = used_pop + [old_pop[i] for i in random_pop]

        new_pop = []

        for i in range(0, len(used_pop)):
            for j in range(i+1, len(used_pop)):
                first_pop = used_pop[i]
                second_pop = used_pop[j]
                item = {
                    "is_image": self.is_image,
                    "pooling": [],
                    "dense": [],
                    "optimizer": model_generator.choose_parameter(
                        first_pop["optimizer"],
                        second_pop["optimizer"],
                        model_generator.get_new_optimizer()
                    ),
                    "last_activation": model_generator.choose_parameter(
                        first_pop["last_activation"],
                        second_pop["last_activation"],
                        model_generator.get_new_activation()),
                    # "epochs": random.randint(10, 1000),
                    "epochs": model_generator.choose_parameter(
                        first_pop["epochs"],
                        second_pop["epochs"],
                        model_generator.get_nb_epochs()),
                    "batch_size": model_generator.choose_parameter(
                        first_pop["batch_size"],
                        second_pop["batch_size"],
                        model_generator.get_batch_size()),
                }

                if self.is_image:
                    for nb_pool in range(model_generator.choose_parameter(
                            len(first_pop["pooling"]),
                            len(second_pop["pooling"]),
                            model_generator.get_nb_pooling())):

                        if nb_pool >= len(first_pop["pooling"]):
                            if nb_pool >= len(second_pop["pooling"]):
                                item["pooling"].append(
                                    model_generator.get_new_random_pooling())
                            else:
                                item["pooling"].append(
                                    model_generator.choose_parameter(
                                        second_pop["pooling"][nb_pool], second_pop["pooling"][nb_pool], model_generator.get_new_random_pooling())
                                )
                        elif nb_pool >= len(second_pop["pooling"]):
                            item["pooling"].append(
                                model_generator.choose_parameter(
                                    first_pop["pooling"][nb_pool], first_pop["pooling"][nb_pool], model_generator.get_new_random_pooling())
                            )
                        else:
                            print("POOLING : {}, {}, {}".format( len(first_pop["pooling"]),
                            len(second_pop["pooling"]), nb_pool))
                            item["pooling"].append(
                                model_generator.choose_parameter(first_pop["pooling"][nb_pool],
                                                                 second_pop["pooling"][nb_pool],
                                                                 model_generator.get_new_random_pooling()))
                max_unit = 200
                for nb_dense in range(model_generator.choose_parameter(
                        len(first_pop["dense"]),
                        len(second_pop["dense"]),
                        model_generator.get_nb_dense())):

                    if nb_dense >= len(first_pop["dense"]):
                        if nb_dense >= len(second_pop["dense"]):
                            item["dense"].append(
                                model_generator.get_new_random_dense(max_unit, self.nb_output))
                        else:
                            item["dense"].append(model_generator.choose_parameter(
                                second_pop["dense"][nb_dense],
                                second_pop["dense"][nb_dense],
                                model_generator.get_new_random_dense(max_unit, self.nb_output)))
                    elif nb_dense >= len(second_pop["dense"]):
                        item["dense"].append(model_generator.choose_parameter(
                            first_pop["dense"][nb_dense], first_pop["dense"][nb_dense], model_generator.get_new_random_dense(max_unit, self.nb_output)))
                    else:
                        item["dense"].append(
                            model_generator.choose_parameter(first_pop["dense"][nb_dense],
                                                             second_pop["dense"][nb_dense], model_generator.get_new_random_dense(max_unit, self.nb_output)))

                new_pop.append(item)
        return new_pop
