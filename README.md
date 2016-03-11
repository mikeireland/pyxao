# pyxao
A python extreme AO library.

The goals of this library is to create a set of AO tools that are:
1) Not too slow. i.e. a reasonable simulation (e.g. a minute of clock time) should be 
able to be completed overnight on a fast desktop.
2) Ability to be parallelized.
3) Understandable an extensible by a senior undergrad student within a couple of weeks 
of effort.
4) Easy to install.
5) Using a popular programming language, preferably a free one for student use.
6) Able to work in the case of extreme-AO.

There are many adaptive optics libraries and codes in existence, so it may seem
silly to be creating a new code here. However, none seem to be extensible easily to fit
these needs. The codes I've found are:

a: COMPASS (http://compass.lesia.obspm.fr). This doesn't seem to be publicly available
(an email didn't go anywhere) and much of this documentation is in French. It also 
appears very complex.

b: OOMAO (https://github.com/rconan/OOMAO). No longer actively supported (see CEO) but 
appears reasonable. Unfortunately in the proprietary MATLAB language.

c: CEO (https://github.com/rconan/CEO). A replacement of OOMAO, which includes GPU 
support. A very complex set of tools, that unfortunately does not include scintillation, 
so is inappropriate for extreme AO and difficult to extend.

d: yao (https://github.com/frigaut/yao). Again - doesn't include scintillation, and is
written in the yao language. Learning this is not a useful skill for the typical 
physics/astro student. 

e: CAOS (https://www-n.oca.eu/caos/). This is written in IDL, which is again proprietary
and not the most useful language for students to learn.

f: soapy (https://github.com/soapy/soapy). Again doesn't include scintillation
(although has the tools to do this) and also appears to be very incomplete and so 
difficult to extend quite yet.

If someone is reading this page and strongly disagrees, please contact the author!

Dependencies are numpy, scipy, astropy and pyfftw. All of these are in anaconda, except
for fftw... which is a bit of a pain to install. 

./configure --enable-shared --enable-float --enable-threads --enable-mpi
make
sudo make install
./configure --enable-shared --enable-threads --enable-mpi
make
sudo make install
./configure --enable-shared --enable-long-double --enable-threads --enable-mpi
make
sudo make install

FFTW really does help - on a quad core i7, it is about a factor of 10 speed up, and 
about 250 double-precision 1024 x 1024 FTs per second. For an 8-10m simulation, an
array size of 512 x 512 (and ~4cm subapertures) is just fine.
