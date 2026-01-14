#!/bin/bash
# parse_args.sh - Reusable argument parser for shell scripts
#
# This script provides a function to parse argument files that may contain:
# - Comments (lines starting with #)
# - Multi-line arguments (e.g., JSON spanning multiple lines)
# - Quoted arguments with spaces
#
# Usage:
#   source parse_args.sh
#   ARGS=()
#   parse_args_file "path/to/args_file.sh" ARGS
#   # Now use "${ARGS[@]}" in your command
#
# Example:
#   # In your script:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/parse_args.sh"
#   ARGS=()
#   parse_args_file "my_args.sh" ARGS
#   my_command "${ARGS[@]}"

# Function to parse a line into arguments, respecting quotes
parse_line() {
    local line="$1"
    local result=()
    local current=""
    local in_quotes=false
    local quote_char=""
    local quote_start_pos=-1
    local i=0
    
    while [[ $i -lt ${#line} ]]; do
        local char="${line:$i:1}"
        
        if [[ "$in_quotes" == false ]]; then
            if [[ "$char" == "'" || "$char" == '"' ]]; then
                in_quotes=true
                quote_char="$char"
                quote_start_pos=${#current}
                # Don't add the opening quote to current
            elif [[ "$char" =~ [[:space:]] ]]; then
                if [[ -n "$current" ]]; then
                    result+=("$current")
                    current=""
                fi
            else
                current+="$char"
            fi
        else
            if [[ "$char" == "$quote_char" ]]; then
                # Check if escaped
                if [[ $i -gt 0 && "${line:$((i-1)):1}" == "\\" ]]; then
                    # Escaped quote, include it
                    current+="$char"
                else
                    # Closing quote - don't add it, just exit quote mode
                    in_quotes=false
                    quote_char=""
                    quote_start_pos=-1
                fi
            else
                current+="$char"
            fi
        fi
        i=$((i + 1))
    done
    
    if [[ -n "$current" ]]; then
        result+=("$current")
    fi
    
    printf '%s\n' "${result[@]}"
}

# Main function to parse an arguments file
# Usage: parse_args_file "path/to/args_file.sh" ARRAY_VAR_NAME
# The parsed arguments will be stored in the array variable specified by ARRAY_VAR_NAME
parse_args_file() {
    local args_file="$1"
    local array_var_name="$2"
    
    if [[ -z "$args_file" ]]; then
        echo "Error: parse_args_file requires an argument file path" >&2
        return 1
    fi
    
    if [[ ! -f "$args_file" ]]; then
        echo "Error: Argument file '$args_file' not found" >&2
        return 1
    fi
    
    if [[ -z "$array_var_name" ]]; then
        echo "Error: parse_args_file requires an array variable name" >&2
        return 1
    fi
    
    # Initialize the array
    eval "$array_var_name=()"
    
    local accumulated=""
    local in_multiline=false
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comment lines (starting with #) and empty lines
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        
        if [[ "$in_multiline" == true ]]; then
            # Continue accumulating multi-line argument
            accumulated="$accumulated"$'\n'"$line"
            # Check if this line closes the JSON (has closing brace followed by quote)
            if echo "$line" | grep -q '}['"'"'"]' || echo "$line" | grep -q '}"' || echo "$line" | grep -q "}'"; then
                # Complete the multi-line argument - replace newlines with spaces for parsing
                accumulated_single_line=$(echo "$accumulated" | tr '\n' ' ')
                while IFS= read -r arg; do
                    if [[ -n "$arg" ]]; then
                        # Properly escape the argument for eval
                        printf -v escaped_arg '%q' "$arg"
                        eval "$array_var_name+=($escaped_arg)"
                    fi
                done < <(parse_line "$accumulated_single_line")
                accumulated=""
                in_multiline=false
            fi
        else
            # Check if this line starts a multi-line JSON argument
            # Pattern: --flag followed by quote and opening brace, but no closing brace+quote on same line
            if echo "$line" | grep -qE "^--[^[:space:]]+[[:space:]]+['\"].*\{" && ! echo "$line" | grep -qE "\}['\"].*$"; then
                accumulated="$line"
                in_multiline=true
            else
                # Single-line argument - parse with proper quote handling
                while IFS= read -r arg; do
                    if [[ -n "$arg" ]]; then
                        # Properly escape the argument for eval
                        printf -v escaped_arg '%q' "$arg"
                        eval "$array_var_name+=($escaped_arg)"
                    fi
                done < <(parse_line "$line")
            fi
        fi
    done < "$args_file"
    
    # Handle any remaining accumulated line
    if [[ -n "$accumulated" ]]; then
        accumulated_single_line=$(echo "$accumulated" | tr '\n' ' ')
        while IFS= read -r arg; do
            if [[ -n "$arg" ]]; then
                # Properly escape the argument for eval
                printf -v escaped_arg '%q' "$arg"
                eval "$array_var_name+=($escaped_arg)"
            fi
        done < <(parse_line "$accumulated_single_line")
    fi
}

