"Borrowed" from https://github.com/cleardusk/3DDFA_V2

I modified the scale of vertices and the deformation basis to be more reasonable.
Meaning the size of the face is now around one unit. And the norm of the basis is also around 1.
This is made "destruction free" by implementing "scaled_*" functions.

Ofc to use these, the original 300w-lp/aflw3d-2000 must be adapted by the inverse scalings.