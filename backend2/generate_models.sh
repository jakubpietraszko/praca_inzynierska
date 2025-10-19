#!/bin/bash

deactivate

source ./.venv/bin/activate

rm -rf ../frontend/src/app/generated-interfaces
mkdir -p ../frontend/src/app/generated-interfaces

for file in interfaces/*.py; do
    if [ -f "$file" ]; then
        module_name=$(basename "$file" .py)
        
        python_module="interfaces.$module_name"
        
        output_file="../frontend/src/app/generated-interfaces/${module_name}.ts"
        
        echo "Generating TypeScript from $python_module to $output_file"
        PYTHONPATH=. pydantic2ts --module "$python_module" --output "$output_file"
    fi
done

deactivate

echo "TypeScript generation completed!"

