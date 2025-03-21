# LLMengine
具体运行步骤可见配套pdf文档中准备工作一栏

这里再重复一波

# steps
```
  1.模型转换，见tools/README.md

  2.将转换后模型的路径，替换到根目录下user_entry.cpp#L5的路径

  3./path/to/LLM-engineering/llama2-7b-tokenizer.bin替换到user_entry.cpp#L6的路径

  4. mkdir build && cd build && cmake .. && make -j8 && ./bin/main
```
