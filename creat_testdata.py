import pandas as pd
import os
import time

# --- 配置 ---
input_folder = '1.0kW/current_csv'  # 原始数据文件夹路径
output_folder = '1.0kW/testdata_csv' # 处理后数据存放文件夹路径
rows_to_keep = 50000                # 需要保留的行数

# --- 主程序 ---

def process_csv_files(input_dir, output_dir, num_rows):
    """
    遍历指定文件夹中的CSV文件，截取指定行数并保存到新文件夹。

    Args:
        input_dir (str): 包含原始CSV文件的文件夹路径。
        output_dir (str): 保存处理后CSV文件的文件夹路径。
        num_rows (int): 每个文件需要保留的前 N 行数据。
    """
    start_time = time.time()
    print(f"开始处理数据...")
    print(f"原始数据文件夹: {input_dir}")
    print(f"目标数据文件夹: {output_dir}")
    print(f"每个文件保留前 {num_rows} 行")

    # 1. 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目标文件夹: {output_dir}")

    # 2. 遍历输入文件夹中的所有文件
    processed_count = 0
    error_count = 0
    for filename in os.listdir(input_dir):
        # 3. 检查文件是否为CSV文件
        if filename.lower().endswith('.csv'):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)

            print(f"\n正在处理文件: {filename} ...")

            try:
                # 4. 使用pandas读取CSV文件的前 N 行
                # nrows 参数可以在读取时就限制行数，效率较高，避免加载整个大文件
                df = pd.read_csv(input_filepath, nrows=num_rows)

                # 检查读取到的行数是否足够（文件本身可能不足50000行）
                if len(df) < num_rows:
                     print(f"  注意: 文件 {filename} 实际行数 ({len(df)}) 少于 {num_rows} 行，已读取所有行。")

                # 5. 将读取到的数据保存到输出文件夹
                # index=False 表示不将pandas DataFrame的索引写入CSV文件
                df.to_csv(output_filepath, index=False, encoding='utf-8') # 通常建议指定utf-8编码

                print(f"  成功处理并将截取后的数据保存至: {output_filepath}")
                processed_count += 1

            except pd.errors.EmptyDataError:
                print(f"  错误: 文件 {filename} 是空的，已跳过。")
                error_count += 1
            except Exception as e:
                print(f"  处理文件 {filename} 时发生错误: {e}")
                error_count += 1
        else:
             print(f"跳过非CSV文件: {filename}")


    end_time = time.time()
    total_time = end_time - start_time
    print("\n--------------------------------------------------")
    print(f"处理完成！")
    print(f"成功处理文件数: {processed_count}")
    print(f"处理失败或跳过文件数: {error_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    print("--------------------------------------------------")

# --- 运行处理函数 ---
if __name__ == "__main__":
    process_csv_files(input_folder, output_folder, rows_to_keep)