# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A simple web-based Tic Tac Toe game contained in a single HTML file with embedded CSS and JavaScript. No build system or dependencies.

## Development

### Running Locally

Serve the static file using Python's built-in HTTP server:

```bash
python3 -m http.server 8080
```

Then open http://localhost:8080/tictactoe.html

### Project Structure

- `tictactoe.html` - Complete game with embedded CSS (styles in `<head>`) and JavaScript (game logic at end of `<body>`)
- `README.md` - Project documentation

### Game Architecture

The game uses a simple state-based approach:
- `gameBoard` array (9 elements) tracks cell states
- `currentPlayer` alternates between 'X' and 'O'
- `winningConditions` array defines all 8 winning combinations (3 rows, 3 columns, 2 diagonals)
- Event listeners on cell buttons handle clicks; disabled attribute prevents re-clicking
- Score persists across games via variables (not localStorage)

## Git Workflow

This repository uses Git and GitHub. Always commit changes with descriptive messages:

```bash
git add <files>
git commit -m "Description of changes"
git push origin main
```

Repository: https://github.com/Tony20221101/tictactoe
