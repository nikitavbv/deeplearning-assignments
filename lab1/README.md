# lab1

Filters where applied to all channels of the image.

## source image

![source](./demo.jpg)

## linear brightness adjustment

![linear brightness adjustment](./report/images/brightness-linear.png)

The image has not changed at all because $H_{min}$ = 0.0$ and $H_{max} = 1.0$.

## exponetial brightness adjustment

![exponential brightness adjustment](./report/images/brightness-exp.png)

## gaussian filter

![gaussian filter](./report/images/gaussian.png)

Side-by-side comparsion with the original image:

![source vs gaussian](./report/images/source-vs-gaussian.png)

## box filter

![box filter](./report/images/box.png)

Side-by-side comparsion with the original image:

![source vs box](./report/images/source-vs-box.png)

## unsharp masking

![unsharp masking](./report/images/unsharp.png)

There are noticable artifacts when you look at the top of the image, but as far as I understand this is expected given how this algorithm works.

## edge detection with sobel operator

![sobel operator](./report/images/edges.png)