## Install MIRTK

Add mirtk to pythonpath

Uses `add2virtualenv` from `virtualenvwrapper`: https://virtualenvwrapper.readthedocs.io/en/latest/command_ref.html#add2virtualenv

```
pip install virutalenvwrapper (virtualenvwrapper-win for windows)
add2virtualenv CMAKE_INSTALL_PREFIX/Lib/Python
```

Then you can directly import mirtk in your python script.

Alternatively, set 
```
export PYTHONPATH=$PYTHONPATH:CMAKE_INSTALL_PREFIX/Lib/Python
```