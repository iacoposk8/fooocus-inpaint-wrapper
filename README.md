
# ComfyUI Fooocus Inpaint Wrapper (Work in progress)
The best inpaint I found is [Fooocus](https://github.com/lllyasviel/Fooocus)'s one, I've never been able to replicate these results on other GUIs. I did something very simple, this is a simple wrapper of practically all the Fooocus code, probably you could do something more refined, probably in the future I'll do something more refined and lightweight but for now it's like this :)

## Install

    cd ComfyUI/custom_nodes
    git clone https://github.com/iacoposk8/ComfyUI-Fooocus-Inpaint-Wrapper
    cd ../..
    .\python_embeded\python.exe -m pip install -r ComfyUI/custom_nodes/ComfyUI-Fooocus-Inpaint-Wrapper/Fooocus/requirements_versions.txt

## How does it work?

If Fooocus is updated in the future, you just need to copy the whole folder and insert it into the node. You can skip copying the models, because it will use the comfyui folder. Inside the copied folder you must also insert the launch.py ​​file that you find in this repository.
You will have to make some changes to the code. Some will be done automatically, you can find them inside the fooocus-inpaint-wrapper.py constructor
These must be done manually for now:
Edit modules/async_worker.py  
and comment on the last line:
#threading.Thread(target=worker, daemon=True).start()  
Then move a few lines up and remove:
while True:  
and
time.sleep(0.01)
