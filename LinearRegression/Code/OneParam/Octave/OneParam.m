#!/usr/bin/octave -q

global theta = 1;
global alpha = 0.1;
global err = 0;

function points = getPoints()

    file_desc = fopen("Data.txt", "r");

    tuple_format = [2 Inf];
    points = fscanf(file_desc,"%d %d", tuple_format);

    fclose(file_desc);

endfunction

function res = h(x)

    global theta;
    res = theta * x;

endfunction

function cost_function(points)

    global err;

    num_points = size(points);
    summary = 0;

    for i=1:num_points(2)
        x = points(1,i);
        y = points(2,i);

        summary += (h(x) - y)^2;
    endfor

    err = summary / (2 * num_points(2));

endfunction

function res = cost_function_derivative(points)

    num_points = size(points);
    summary = 0;

    for i=1:num_points(2)
        x = points(1,i);
        y = points(2,i);

        summary += (h(x) - y) * x;
    endfor

    res = summary / num_points(2);

endfunction

function gradient_decend(points)

    global theta alpha;
    theta = theta - alpha * cost_function_derivative(points);

endfunction

points = getPoints();

printf("\nalpha = %f\n\n", alpha);

for i=1:3
    cost_function(points);
    printf("theta = %f \t error = %f\n", theta, err);
    gradient_decend(points);
end
