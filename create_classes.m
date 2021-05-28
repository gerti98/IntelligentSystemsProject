function [y_classes] = create_classes(y_vector, strange)
    samples_vect = size(y_vector);
    samples = samples_vect(2);
    y_classes = zeros(7, samples);
    classes_types = [1 strange 11/3 5 19/3 23/3 9];


    for i=classes_types
        indexes = find(y_vector==i);
        y_classes(find(classes_types==i), indexes) = 1
    end
end