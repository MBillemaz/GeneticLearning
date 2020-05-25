from ga import Genetic_algo
from model_generator import model_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
import matplotlib.pyplot as plt
import gc

class GeneticDL:
    def __init__(self, nb_output):
        self.nb_output = nb_output

    @tensorflow.autograph.experimental.do_not_convert
    def image_train(self, path):
        self.ga = Genetic_algo(self.nb_output, True)
        datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            validation_split=0.2,
            rotation_range=30,
            shear_range=0.15
        )
        class_mode = "categorical"
        if self.nb_output == 1:
            class_mode = "binary"
        population = self.ga.get_first_pop(10)
        
        training_set = datagen.flow_from_directory(
            path,
            target_size=(64, 64),
            batch_size=20,
            class_mode=class_mode,
            subset="training"
        )

        test_set = datagen.flow_from_directory(
            'dataset',
            target_size=(64, 64),
            batch_size=20,
            class_mode=class_mode,
            subset="validation"
        )
        for gen in range(20):
            
            scored_pop = []
            for pop in range(len(population)):
                #try:

                model = model_generator(population[pop], self.nb_output)

                classifier = model.fit(
                    training_set,
                    steps_per_epoch=training_set.samples,
                    epochs=population[pop]['epochs'],
                    validation_data=test_set,
                    validation_steps=test_set.samples,
                    # use_multiprocessing=False,
                    workers=8
                )
                print("Score Gen {} pop {} : {}".format(gen, pop, classifier.history["val_accuracy"][-1]))
                print("-----------------------------------------------")
                scored_pop.append({"pop": population[pop], "model": model, "score": classifier.history["val_accuracy"][-1]})
                
               # except:
               #     print("Error during model generation")
               #     scored_pop.append({"pop": pop, "model": None, "score": 0})
            
            scored_pop.sort(key=lambda x: x["score"], reverse=True)
            scored_pop[0]["model"].save("models/Gen{}_{}".format(gen + 1, scored_pop[0]["score"]))
            
            ordered_pop = [item["pop"] for item in scored_pop]
            
            plt.plot(range(1, 11), [item["score"] for item in scored_pop], label="Gen {}".format(gen))
            
            if (gen+1)%5 == 0:
                plt.xlabel('Model')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig("plots/Gen_{}.png".format(gen), format="png")
                plt.clf()
            
            gc.collect()
            
            population = self.ga.new_generation(ordered_pop)

    # def new_pop(is_image):
        
