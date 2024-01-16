function[out]=normalize_2D(in)
out=(in-min(min(in)))/max(max(in-min(min(in))));