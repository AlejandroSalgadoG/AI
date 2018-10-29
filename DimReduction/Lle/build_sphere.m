function [x y z] = build_sphere(n, r)
    angle = rand(n,1) * 360;
    radius = sqrt(rand(n,1)) * r;
    height = ones(n,1)*r;
    
    x = arrayfun(@cos, angle).*radius;
    y = arrayfun(@sin, angle).*radius;
    z = sqrt( height.^2 - radius.^2);
end