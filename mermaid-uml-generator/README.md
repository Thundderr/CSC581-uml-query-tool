# GitHub Repository UML Class Diagram Generator (Using Mermaid)

A Python script that gets the top 10 most starred public repositories with a specified programming language and license and generates Mermaid UML Class diagrams, both in text form (.mmd) and vector form (.svg).

**As of now, only repositories in Python and Java are supported.**

## Requirements

- **Python 3.12**
- **Node.js** (for Mermaid installation)

## Usage
Ensure that node.js is installed:
```bash
node -v
```

If node.js is missing, install from https://nodejs.org

Install Mermaid CLI (For images):
```bash
npm install -g @mermaid-js/mermaid-cli
```

Verify that Mermaid is installed:
```bash
mmdc -h
```

Set a GitHub personal access token with `repo:public_repo` or at least read access to public repos and no special scopes. Export it as `GITHUB_TOKEN`:
```bash
export GITHUB_TOKEN="ghp_...."
```

Now, in your Python environment:
```bash
# Follow directions on the main repository page

# CD into this folder
cd ./mermaid-uml-generator

# Install dependencies
pip install -r requirements.txt
```

### Mermaid CLI SVG generation

If you want the script to automatically convert your `.mmd` files to `.svg` files:

Create a .json file in this folder named `puppeteer-config.json`.
***Mermaid requires Chromium to render the .mmd files.*** 
```json
{
  "executablePath": "your/chrome/path/here",
  "args": [
    "--no-sandbox",
    "--disable-setuid-sandbox"
  ]
}
```

Once you have these set up, you should be able to run the below command without issues:

```bash
# Example command, refer to the script to see what arguments you can pass.
python3 ./repo-uml-generator.py --language python --license mit
```

### Dataset Location

By default (unless you have passed a different --out argument), the script should save the diagrams in the `dataset_out` folder.

## Structure

```
mermaid-uml-generator/
├── repo-uml-generator.py   # Main script
├── dataset_out/            # Created once script runs
│   ├── repo-name-01/
│   ├── repo-name-02/
│   ├── repo-name-03/
│   └── repo-name-04/
├── puppeteer-config.json   # Configuration file
└── requirements.txt
```

## License

This project is for educational purposes (CSC581). All repositories viewed by this script should have a license that allows for Mermaid UML generation:

```
"MIT",
"Apache-2.0",
"BSD-2-Clause",
"BSD-3-Clause",
"ISC",
"CC0-1.0"
```
