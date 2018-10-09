function [x y z]= build_swiss_roll(n)
    t = 3*pi/2 * (1 + 2*rand(n,1));
    x = t.*cos(t);
    y = 5 * rand(n,1);
    z = t.*sin(t);
end