## Setup Instructions

Follow the steps below to set up the environment and start training the models.

### Step 1: Unzip the Project

Unzip the project files into a directory on your machine:
```bash
unzip submission.zip
cd submission/code
```

### Step 2: Create venv

Set up a virtual env to manage dependencies:
```bash
python -m venv env
source venv/bin/activate  # mac
env\Scripts\activate     # win
```

### Step 3: Install Dependencies

Make sure you have all packages by running:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Code

Start the training by running:
```bash
python train.py
```

### Step 5: Run the Code

Once the model has been trained
```bash
python test.py
```