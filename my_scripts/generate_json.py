#!/usr/bin/env python3
import os
import json

def main():
    base_dir = '/mnt/data/yangzengzhi/data/renders_cond'
    # 收集所有子目录名
    names = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    names.sort()

    # 输出 JSON
    out_path = '../model_names_list.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(names, f, indent=2, ensure_ascii=False)

    print(f'Found {len(names)} folders, wrote to {out_path}')

if __name__ == '__main__':
    main()