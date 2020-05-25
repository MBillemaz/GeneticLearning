
from GeneticDL import GeneticDL

test = GeneticDL(6)
test.image_train("dataset")
#
#datagen = ImageDataGenerator(
#    rescale = 1./255,
#    horizontal_flip=True,
#    validation_split=0.2,
#    rotation_range=30,
#	shear_range=0.15
#    )
#
#population = ga.get_first_pop(10, True)
#
#for gen in range(1):
#    
#    scored_pop = []
#    for pop in range(len(population)):
#        try: 
#            training_set = datagen.flow_from_directory(
#                'dataset',
#                target_size=(64,64),
#                batch_size=20,
#                class_mode="categorical",
#                subset="training"
#                )
#        
#            test_set = datagen.flow_from_directory(
#                'dataset',
#                target_size=(64,64),
#                batch_size=20,
#                class_mode="categorical",
#                subset="validation"
#                )
#            model = model_generator(population[pop], 6)
#            
#            classifier = model.fit_generator(
#                training_set,
#                steps_per_epoch=training_set.samples,
#                epochs=population[pop]['epochs'],
#                validation_data=test_set,
#                validation_steps=test_set.samples,
#                use_multiprocessing=False,
#                verbose=1,
#                workers=8
#                )
#
#            classes = training_set.class_indices
#            classes = {v: k for k, v in classes.items()}
#
#
#            score = 0
#            for name in ['Cavalier', 'Fou', 'Pion', 'Reine', 'Roi', 'Tour']:
#                
#                test_image= image.load_img("validation/{}.jpg".format(name), target_size=(64,64))
#                
#                # =============================================================================
#                # test_image2= image.load_img("dataset/single_prediction/cat_or_dog_2.jpg", target_size=(64,64))
#                # =============================================================================
#                
#                test_image = image.img_to_array(test_image)
#                test_image = np.expand_dims(test_image, axis=0)
#                
#                
#                result = classifier.predict_classes(test_image)
#
#                if(name == classes[result[0]]):
#                    score += 1
#                
#            scored_pop.append([population[pop], score])
#                
#            print('précision : {}'.format(score/6))
#        except Error:
#            print("Error : {}".format(Error))
#            scored_pop.append([population[pop], 9999])
#            
#            
#%%
