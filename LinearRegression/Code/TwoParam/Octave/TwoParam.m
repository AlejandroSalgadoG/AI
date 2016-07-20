#!/usr/bin/octave -q

global theta_0 = 1;
global theta_1 = 1;
global alpha = 0.1;
global err = 0;

function points = getPoints()

    file_desc = fopen("Data.txt", "r");

    tuple_format = [2 Inf];
    points = fscanf(file_desc,"%d %d", tuple_format);

    fclose(file_desc);

endfunction

function res = h(x)

    global theta_0 theta_1;

    res = theta_0 + theta_1 * x;

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

function res = cost_function_derivative_theta_0(points)

    num_points = size(points);
    summary = 0;

    for i=1:num_points(2)
        x = points(1,i);
        y = points(2,i);

        summary += (h(x) - y);
    endfor

    res = summary / num_points(2);

endfunction

function res = cost_function_derivative_theta_1(points)

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

    global theta_0 theta_1 alpha;
    temp_0 = theta_0 - alpha * cost_function_derivative_theta_0(points);
    temp_1 = theta_1 - alpha * cost_function_derivative_theta_1(points);

    theta_0 = temp_0;
    theta_1 = temp_1;

endfunction

points = getPoints();

printf("\nalpha = %f\n\n", alpha);

for i=1:3
    cost_function(points);

    printf("theta_0 = %f\n", theta_0);
    printf("theta_1 = %f\n", theta_1);
    printf("error = %f\n", err);
    printf("\n");

    gradient_decend(points);
end
