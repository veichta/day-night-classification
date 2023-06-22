# day-night-classification
Easy to use classification model trained to classify day and night images.

# Install
Make sure to install all requirements before running the code.
```
pip install -r requirements.txt
```

Installing ```dnc``` package:
```
pip install -e .
```

# Usage (Python)
In order to infer on an image:
```
import numpy as np
from dnc.inference import infer

img = <np.ndarray>
infer(img)
```

Or directly from a file:
```
from dnc.inference import infer_from_file

infer_from_file(<path_to_image>)
```


# Usage (Command Line)
To infer on an image:
```
python dnc/inference.py --image_path <path_to_image>
```

For more information on the arguments:
```
python dnc/inference.py --help
```

# Training
To train the model:
```
python dnc/train.py \
    --data_dir <path_to_data_dir> \
    --night_images <name_of_night_image_folder> \
    --day_images <name_of_day_image_folder> \
    --device <device_to_train_on> \
```

For more information on the arguments:
```
python dnc/train.py --help
```
