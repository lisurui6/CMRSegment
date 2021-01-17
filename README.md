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


```
docker build -t lisurui6/cmr-segment:latest .
docker push docker.io/lisurui6/cmr-segment:latest
```

```
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa
```

```
export PYTHONPATH=$PYTHONPATH:/mirtk/lib/python
rf -rf PH_atlas
apt install python3.7
pip2 uninstall tensorflow
python3.7 get-pip.py
pip3.7 install -r requirements.txt
pip3.7 install tensorflow
pip3.7 install torch torchvision
```