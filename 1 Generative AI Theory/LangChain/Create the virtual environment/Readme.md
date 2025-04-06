
---

## ✅ Conda Environment Setup (Using Custom Path)

### 📌 Step 1: Open Command Prompt (CMD)

You can do this by:
- Pressing `Win + R`, typing `cmd`, and hitting Enter
- Or using the **Terminal in VS Code**, set to `cmd`

---

### 📌 Step 2: Create the Conda Environment in Your Project Folder

Navigate to your project folder, e.g.:

```cmd
cd E:\LangChain
```

Create the environment (Python 3.10 used as an example):

```cmd
conda create -p venv python=3.10 -y
```

This creates the environment inside the `venv` folder **in your current directory**.

---

### 📌 Step 3: Initialize Conda for CMD (One-time Setup)

If you haven’t already done this:

```cmd
conda init cmd.exe
```

Then **close and reopen CMD**.

---

### 📌 Step 4: Activate the Environment

From inside your project folder:

```cmd
conda activate .\venv
```

Or from anywhere:

```cmd
conda activate E:\LangChain\venv
```

---

### 📌 Step 5: Deactivate the Environment

To deactivate:

```cmd
conda deactivate
```

---

## 🧠 Bonus: Use It in VS Code

1. Open VS Code in your project folder (`E:\LangChain`)
2. Press `Ctrl+Shift+P` and search for:
   ```
   Python: Select Interpreter
   ```
3. Choose:
   ```
   E:\LangChain\venv\python.exe
   ```

If it’s not visible, click:
> `Enter interpreter path` → `Browse...` → select  
> `E:\LangChain\venv\python.exe`

---
