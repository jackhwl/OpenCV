## Section 2: Installing Jupyter Notebook
* jypyter notebook --port=8889 --ip=127.0.0.1 --notebook_dir=notebook_path --no-browser
* docker pull jupyter/datascience-notebook
* docker run -it --rm -p 8888:8888 -v c:\users\jack\notebooks:/home/jovyan jupyter/datascience-notebook
## Section 3: Moving from the REPL to a Notebook
```
class Person(object):
    def __init__(self, first, last):
        self.first_name = first
        self.last_name = last
    def comma_name(self):
        return '{0}, {1}'.format(self.last_name, self.first_name)
```
* !, _, panda
## Section 4: Leveraging Special Notebook Features
* X C V Shift-V D,D A B
* docstring ?