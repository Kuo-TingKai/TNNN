# Folder Tree (Current)
- TNML_2022
    - data
    - history
        - model_arch
            - timestamp
                - acc
                - loss
                - val_acc
                - val_loss
    - img
    - model_weight
        - model_arch
            - timestamp
                - model_weight
    - **src**
        - relic
        - **util**
            - save_history.py
            - save_load_model.py
            - timestamp.py
        - model.py (to be divided by multiple parts)
        - mpo.py
        - resize.py (to be move to util)
        - save_dataset.y (to be move to util)

# Instruction
- Run model silently: 
```bash
nohup python3 ./src/{model}.py model_architecture nepoch bond_dim nnode dataset &
```

# Some tips

- Using `tmux` in wsl-2 can divided window into multiple panes
- pdfcrop --margin 5  {source}.pdf {output}.pdf
- hdf5 viewer

# TODO

- md2pdf cannot convert latex
