import os
import selection
import variation
import prediction


def run_each_generation(program_list, generation_count, df, gap, tdp, MAX_PROGRAM_SIZE,
                        dataset, NUMBER_OF_REGISTERS, vr_obj, data_module_mapping, t, rp, st):
    X_train, y_train, X_test, y_test = train_test_split(df, tdp, dataset)

    # Scaling(if required)
    X_train, X_test = data_module_mapping[dataset].get_scaled_data(
        X_train, X_test)

    # # print(pd.get_dummies(y_train))
    unique_classes = sorted(list(set(y_train)))
    register_class_map = {
        "R"+str(i): unique_classes[i] for i in range(len(unique_classes))}

    # print(register_class_map)
    train_accuracy = []
    fitness_scores = {}

    # Put training function here
    train_accuracy, program_list = data_module_mapping[dataset].train_gp_classifier(generation_count, program_list, train_accuracy, fitness_scores,
                                                                                    vr_obj, X_train, y_train, register_class_map, gen, gap, dataset,
                                                                                    MAX_PROGRAM_SIZE, t, rp, st
                                                                                    )

    # Check, save and print metrics of best classifier
    save_and_test_champ_classifier(program_list, X_train, y_train, X_test, y_test, register_class_map,
                                   NUMBER_OF_REGISTERS, dataset, st)
