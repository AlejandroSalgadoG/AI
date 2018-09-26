function data = build_swiss_roll(n)
    t = 3*pi/2 * (1 + 2*rand(n,1));
    h = 5 * rand(n,1);
    data = [t.*cos(t), h, t.*sin(t)];
end