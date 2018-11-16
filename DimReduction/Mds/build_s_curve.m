function [x y z] = build_s_curve(n)
    t = 3 * pi * (rand(n,1) - 0.5);
    x = sin(t);
    y=2 * rand(n,1);
    z = t./abs(t) .* (cos(t) - 1);
end