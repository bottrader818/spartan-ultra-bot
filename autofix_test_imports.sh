#!/bin/bash
set -euo pipefail

# Enhanced pytest auto-import fixer with debugging and cleanup

# Configuration
TARGET_DIR="./tests"  # Directory to scan for test files
IMPORT_SEARCH_DIRS=("." "./src" "./lib")  # Directories to search for import sources
EXCLUDE_DIRS=("venv" ".venv" "__pycache__" ".git" ".mypy_cache" ".pytest_cache")  # Directories to exclude
PY_EXT=(".py")  # Python file extensions to process
DEBUG=${DEBUG:-false}  # Set DEBUG=true for verbose output
BACKUP_SUFFIX=".bak"  # Suffix for backup files
CLEANUP=${CLEANUP:-false}  # Set CLEANUP=true to remove backup files

# Debug logging function
debug() {
    if [[ "$DEBUG" == "true" ]]; then
        echo "DEBUG: $1"
    fi
}

# Cleanup function
cleanup() {
    if [[ "$CLEANUP" == "true" ]]; then
        echo "🧹 Cleaning up backup files..."
        find "$TARGET_DIR" -name "*$BACKUP_SUFFIX" -exec rm -v {} +
    fi
}

# Register cleanup on exit
trap cleanup EXIT

echo "🔍 Scanning pytest output for import issues..."
debug "Configuration:"
debug "  TARGET_DIR: $TARGET_DIR"
debug "  IMPORT_SEARCH_DIRS: ${IMPORT_SEARCH_DIRS[*]}"
debug "  EXCLUDE_DIRS: ${EXCLUDE_DIRS[*]}"
debug "  DEBUG: $DEBUG"
debug "  CLEANUP: $CLEANUP"

# Get undefined names from pytest output
debug "Running pytest to detect undefined names..."
pytest_output=$(pytest 2>&1 || true)
undefined_names=$(echo "$pytest_output" | grep -oE "name '[A-Za-z0-9_]+' is not defined" | cut -d"'" -f2 | sort -u)

if [[ -z "$undefined_names" ]]; then
    echo "✅ No undefined names found in pytest output"
    exit 0
fi

echo "⚠️  Found undefined names:"
echo "$undefined_names" | sed 's/^/  - /'
debug "Full pytest output:\n$pytest_output"

# Function to find potential import path for a name
find_import_path() {
    local name=$1
    local found_path=""
    
    debug "Searching import path for: $name"
    
    # First check if it's a standard library module
    if python -c "import $name" 2>/dev/null; then
        debug "Found in standard library: $name"
        echo "import $name"
        return 0
    fi
    
    # Search through project directories
    for dir in "${IMPORT_SEARCH_DIRS[@]}"; do
        if [[ ! -d "$dir" ]]; then
            debug "Skipping non-existent directory: $dir"
            continue
        fi
        
        debug "Searching in directory: $dir"
        
        # Build find command with all exclude patterns
        local find_cmd="find '$dir' -type f -name '*.py'"
        for exclude in "${EXCLUDE_DIRS[@]}"; do
            find_cmd+=" ! -path '*$exclude*'"
        done
        
        # Find files containing the name as class/function definition
        while IFS= read -r -d '' file; do
            debug "Checking file: $file"
            
            # Check if the name is defined in the file
            if grep -q -E "^(class|def|async def) $name\b" "$file"; then
                local rel_path="${file#$dir/}"
                rel_path="${rel_path%.py}"
                rel_path="${rel_path//\//.}"
                debug "Found definition in: $rel_path"
                echo "from $rel_path import $name"
                return 0
            fi
        done < <(eval "$find_cmd -print0" 2>/dev/null)
    done
    
    debug "Could not find import path for: $name"
    return 1
}

# Process each undefined name
echo "$undefined_names" | while read -r name; do
    echo -e "\n⚙️  Processing: $name"
    debug "Starting processing for: $name"
    
    # Find where this name should be imported from
    import_statement=$(find_import_path "$name") || true
    
    if [[ -z "$import_statement" ]]; then
        echo "❌ Could not determine import path for: $name"
        debug "Consider adding manual import for: $name"
        continue
    fi
    
    echo "📌 Recommended import: $import_statement"
    debug "Using import statement: $import_statement"
    
    # Build grep exclude pattern
    local grep_exclude=""
    for exclude in "${EXCLUDE_DIRS[@]}"; do
        grep_exclude+=" --exclude-dir=$exclude"
    done
    
    # Find files that need this import
    file_count=0
    while IFS= read -r -d '' file; do
        debug "Processing file: $file"
        
        # Skip if the import already exists
        if grep -q -E "^${import_statement% *}" "$file"; then
            echo "🔍 Import already exists in: ${file#$TARGET_DIR/}"
            debug "Skipping file as import exists: $file"
            continue
        fi
        
        echo "✏️  Adding to: ${file#$TARGET_DIR/}"
        
        # Backup original file
        backup_file="$file$BACKUP_SUFFIX"
        debug "Creating backup: $backup_file"
        cp "$file" "$backup_file"
        
        # Insert import below the last existing import line
        debug "Modifying file: $file"
        awk -v import_stmt="$import_statement" '
            BEGIN { inserted = 0; last_import_line = 0 }
            /^from |^import / {
                print
                last_import_line = NR
                next
            }
            /^[^#]/ && !inserted && (NR > last_import_line) && (last_import_line > 0) {
                # Insert after last import but before next non-import, non-comment line
                print import_stmt
                inserted = 1
            }
            {
                print
            }
            END {
                if (!inserted && last_import_line == 0) {
                    # No imports found, add at top
                    print import_stmt
                }
            }
        ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
        
        ((file_count++))
        debug "Successfully modified file: $file"
    done < <(grep -rln --include="*.py" $grep_exclude "\b$name\b" "$TARGET_DIR" 2>/dev/null | tr '\n' '\0')
    
    if [[ $file_count -eq 0 ]]; then
        echo "ℹ️  No files needed modification for $name"
    else
        echo "✔️  Updated $file_count files for $name"
    fi
done

echo -e "\n✅ Auto-fix complete. Recommendations:"
echo "1. Review all changes before committing (use 'git diff' or similar)"
echo "2. Run: pytest to verify fixes"
echo "3. Check for *$BACKUP_SUFFIX files if you need to revert changes"
echo "4. To clean up backup files, run with CLEANUP=true"
echo "5. For debugging output, run with DEBUG=true"

# Run pytest again to show fixed status
echo -e "\nRunning pytest to verify fixes..."
pytest || echo "⚠️  Some tests may still be failing - please review manually"
