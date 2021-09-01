---
title:  去年阅读 《C++ Primer 中文版(第5版)》 的印象笔记
date: 2021-05-23 17:40:10
tags:
    - C++ 编程学习
categories: 杂七杂八
toc: true
---

去年5月份我一个人从上海跳槽来广州，在这半年多的独居生活里我利用零碎的时间读了这本久负盛名 《C++ Primer 中文版(第5版)》。当时每周双休的时候坚持读大概二十页，竟不知不觉也读了一半多。但我觉得还是第六版写得更好，否则我也不会读了七八遍。而这本书还尚未读完，惭愧！

<p align="center">
    <img width="40%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/C++_Primer-第5版-阅读笔记-20210524180315.jpg">
</p>

今天整理文档的时候发现了这份宝藏笔记，于是乎拿出来晒一晒。

<!-- more -->




第一章 开始
===========
编写一个简单的 C++ 程序
---------------------------
每一个 C++ 程序都包含一个或多个函数(function)，但是其中一个必须命名为 `main`，操作系统将调用 `main` 来运行 C++ 程序。下面是一个非常简单的 `main` 函数，它什么也不干，只是返回操作系统一个值：

```cpp
int main(){
  return 0;
}
```

大多数系统中，`main` 的返回值被用来指示状态。返回值 0 表明成功，非 0 的返回值的含义由系统来定义，通常用来指出错误类型。然后我们可以用 `gcc` 对它进行编译和运行:

```bashrc
$ g++ prog1.cc
$ ./a.out
```

我们可以通过 `echo` 来获取其返回值，在 `UNIX` 系统中，通过如下命令获取状态:

```bashrc
% echo $?
0
```

初识输入输出
----------------
`iostream` 库包括两个基础类型 `istream` 和 `ostream`，分别表示输入流和输出流。
### 向流写入数据

```cpp
std::cout << "Enter two numbers:";
std::cout << std::endl;
```

第一个输出运输符给用户打印一条信息，在双引号之间的文本将被打印到标准输出。第二个运算符打印 `endl` ，它表示的效果是结束当前行，并将设备关联的缓冲区（buff) 中的内容刷新到设备中，从而可以保证目前为止程序所产生的所有输出都真正写入到输入流中，而不是仅停留在内存中等待写入流。
### 从流读取数据

```cpp
#include <iostream>

int main(){
    int i, j;
    std::cin >> i >> j;
    std::cout << i + j << std::endl;
    return 1;
}
```

编译运行后：

```
    $ ./a.out
    1
    2
    3
```

第二章 变量和基本类型
=====================
基本内置类型
----------------
通过添加下表所列的前缀和后缀，可以改变整型，浮点型和字符型字面值的默认类型：

-   对于字符和字符串字面值

  |前缀|含义|类型|
  |---|---|---|
  |u|      Unicode 16 字符|   char16_t
  |U|      Unicode 32 字符|   char32_t
  |L|      宽字符|            wchar_t
  |u8|     UTF-8|             char

-   对于整型字面值

  |后缀|       最小匹配类型|
  |---|---|
  |u or U|     unsigned
  |l or L|     long
  |ll or LL|   long long

-   对于浮点型字面型

  |后缀|     类型
  |---|---|
  |f or F|   float
  |l or L|   long double

例如：

    L'a'          // 宽字符型字面值，类型是 wchar_t
    u8"hi!"       // utf-8 字符串字面值
    42ULL         // 无符号整型字面值， 类型是 unsigned long long
    1E3-F         // 单精度浮点型字面值，类型是 float
    3.14159L      // 扩展精度浮点字面型，类型是 long double

变量
--------
变量定义的基本形式是：首先是类型说明符，随后紧跟由一个或多个变量名组成的列表，其中变量名以逗号分割，最后以分号结束。例如

```cpp
int sum = 0, val = 0;
double price = 109.9, discount = price * 0.16;
```

变量都是具有作用域的，一般用花括号 `{ }` 去声明，有时候会配合关键词 `namespace`，例如:

```cpp
namespace cv{
    int val = 0;
    int sum = 0;
};
```

上述这种带名字的作用域既可以出现在 `.h` 文件中，也可以出现在 `.cpp`
文件中。

复合类型
------------
复合类型是基于其他类型定义的类型。C++ 语言有几种复合类型，本章将介绍其中的两种：引用和指针。
### 引用
引用（reference）为对象起了另外一个名字，引用类型引用另外一种类型。通过将声明符写成
`&d` 的形式来定义引用类型，其中 `d` 是声明<strong>变量的别名</strong>：

```cpp
int ival = 1024;
int &refVal = ival;             // refVal 是 ival 的另外一个名字
int &refVal2;                 // 报错，引用必须初始化
```

允许在一条语句中定义多个引用，其中每个引用标识符都必须以符号`&`开头：

```cpp
int i=1024, j=2048;
int &ri=i, &rj=j;
```

### 指针
#### 定义
与引用类似，指针也实现了对其他对象的间接访问，然而指针与引用又有很多不同点。其一，指针本身就是一个对象，允许对指针赋值和拷贝，而且在指针的生命周期内他可以先后指向几个不同的对象。其二，指针无须在定义时赋初值。

```cpp
int *ip1, *ip2;    // ip1 和 ip2 都是指向 int 型对象的指针
```

指针存放某个对象的地址，要想获取该地址，需要使用取地址符号 `&`：

```cpp
int ival = 42;
int *p = &ival;     // p 存放的是 ival 的地址
```

如果指针指向了一个对象，则允许使用解引用符 `*` 来访问该对象：

```cpp
int ival = 42;
int *p = &ival;
cout << *p;         // 由 * 得到指针 p 所指向的对象，输出 42
```

#### nullptr 和 void 指针
有两种指针需要特别注意：

-   nullptr 指针：nullptr 是 C++11 新标准刚引入的一种方法，它可以被转换成任意其他类型的指针；
-   void
    指针：这是一种特殊的指针类型，可用于存放任意对象的地址，这就很牛逼了。

```cpp
double obj = 3.14, *pd = &obj;
void *pv = &obj                 // obj 可以是任意类型的对象
pv = pd;                        // pv 是可以存放任意类型的指针
```

#### 指向指针的指针
指针是内存的对象，像其他对象一样也有自己的地址，因此允许把<strong>指针的地址再存放到另一个指针当中。</strong>通过 `*` 的个数可以区分指针的级别，也就是说，`**` 表示指向指针的指针，`***` 表示指向指针的指针的指针，以此内推：

```cpp
int ival = 1024;
int *pi = &ival;     // 一级指针 pi，它指向一个 int 的数
int **ppi = &pi;     // 二级指针 ppi，它指向一个 int 型的指针
```

二级指针在实际工作中有很多用处，例如我在工作中就遇到过这样的代码：

```cpp
IOnnxSession *CreateOnnxSession(const wchar_t *model_path, int num_threads, int gpu_id) {
    return new OnnxSession(model_path, num_threads, gpu_id);
}

void ReleaseOnnxSession(IOnnxSession **ptr) { // ptr 是一个二级指针
    if (ptr && *ptr) {
        delete *ptr;
        *ptr = NULL;
    }
}

const wchar_t *model_path = "yolo.onnx";
yolo_session = CreateOnnxSession(model_path, 1, 0);  // 创建一个指向 IOnnxSession 的一级指针
ReleaseOnnxSession(&yolo_session);                   // 释放内存
```

其实这里有个非常值得深思的地方，为什么用于销毁对象的 `ReleaseOnnxSession` 函数的形参要用一个二级指针呢，一级指针不行吗？我们来看看，假如我们用一级指针的话，那么该函数应该这样改写

```cpp
void ReleaseOnnxSession(IOnnxSession *ptr) { // ptr 是一个一级指针
    if (ptr) {
        delete ptr;
        ptr = NULL;
    }
}
ReleaseOnnxSession(yolo_session);                   // 释放内存
```

看起来好像没什么问题，但其实这样做的安全隐患很大：指针 `yolo_session` 传入函数 `ReleaseOnnxSession` 中会<strong>发生拷贝，从而产生一个临时变量指针</strong> `_yolo_ssesion`，由于 `_yolo_session` 存放的地址和 `ptr` 是一样的，因此 `delete` 操作能够把对象 `yolo_session` 给销毁掉。但是却不会把 `yolo_ssesion` 指针置为 `nullptr` 指针，这就是使得 `yolo_session` 成了一个[野指针](https://baike.baidu.com/item/野指针)。

限定符
----------------
-   因为 const 对象一旦创建后就不能改变，<strong>所以 const 对象必须初始化</strong>；
-   默认状态下，const 对象仅在文件内有效，如果需要被其他文件访问，则需要加 extern；

### const 的引用
与普通引用不同的是，const 引用不能通过别名去改变它所绑定的对象。

```cpp
int ci = 1024;
const int &r1 = ci;     // 不能通过别名 r1 去改变 ci 的值
```

但是这并不意味着 ci 的值就不能改变了，只是不能通过 r1 去改变而已。

### 指针和 const
如果我们要想存放常量对象的地址，就必须使用指向常量的指针：

```cpp
const double pi = 3.14;     // pi 是个常量，它的值不能改变
double *ptr = &pi;          // 错误，普通指针不能存放常量的地址
const double *cptr = &pi;   // 正确
```

还有一种指针是常量性的，即 const 指针：常量指针一旦初始化，就不能改变它的值了（即存放在指针中的那个地址），即不变的是指针本身而非指向的那个值。

```cpp
int val = 10;
int *const ptr = &val;      // 以后就不能改变 ptr 所指向的地址了
```

这里值得注意的是，指针本身是一个常量并不意味着就不能通过该指针修改所指向对象的值。能否这样做完全取决于所指对象的类型，例如：

```cpp
double pi = 3.14;
const double *const pip = &pi; // 这里既不能改变指针 pip 的值，也不能通过 pip 去改变 pi 的值
```

### 顶层 const
指针本身是一个对象，它又可以指向另一个对象。因此，指针本身是不是常量以及指针所指向的是不是常量就是两个相互独立的问题。用名词**顶层 const** 表示指针本身就是个常量，而用名词**底层 const** 表示指针所指向的对象是一个常量。

```cpp
int i = 0;
int* const p1 = &i;       // 不能改变 p1 的值，这是一个顶层 const
const int ci = 42;
const int *p2 = &ci;      // p2 所指的对象是一个常量，而且 p2 可以改变，所以这是一个底层 const
```

### constexpr 和常量表达式
**常量表达式（const expression)** 是指值不会改变并且在**编译过程**中就能得到计算结果的表达式。C++规范在一些地方要求使用常量表达式，如声明数组的维数，以下这样是错误的：

```cpp
int get_five() {
    return 5;
}
int data[get_five() + 7]; // 创建包含12个整数的数组. 这是非法的，因为get_five() + 7不是常量表达式
```

在实际中，我们很难判断是不是常量表达式，幸运的是 C++11 引入了关键字 constexpr，允许编程者保证函数或对象的构造函数是编译时常量。上述代码可以改写为：

```cpp
constexpr int get_five() {
    return 5;
}

int data[get_five() + 7]; // Create an array of 12 integers. Valid C++11
```

非常需要注意的是 constexpr 关键词是为了在编译期间进行优化用的，如果相关参数在运行期间才能确定的话，就不要用它了。

处理类型
------------
有时候一些类型的名字难以拼写，我们无法明确体现其真实目的和含义。这时候如果用类型别名，就能很好帮助我们理解类型的语义信息。

### 类型别名
有两种方法定义类型别名。传统的方法是使用关键字 `typedef`:

```cpp
typedef double wages;       // wages 是 double 的同义词
typedef wages base, *p;     // base 是 double 的同义词，p 是 double* 的同义词
```

`C++11` 新标准规定了另一种新方法，使用别名声明来定义类型的别名:

```cpp
using SI = Sales_item;    // SI 是 Sales_item 的同义词
```

### auto 类型说明符
编程时常常需要把表达式的值赋给变量，这就要求在声明变量的时候清楚地知道表达式的类型，然而做到这一点并非那么容易。为了解决这个问题，C++11 新标准引入了 `auto` 类型说明符，用它就可以让编译器替我们去分析表达式所属的类别。

```cpp
int val1 = 10, val2 = 20;
auto val3 = val1 + val2;       // 自动推断 val3 的类型为 int
```

使用 `auto` 也能在一条语句中声明多个变量:

```cpp
auto i=0, *p=&i;               // 正确: i 时整数，p 是整型指针
```

### decltype 类型指示符
`auto` <strong>有个鸡肋的地方是，它定义的变量必须要初始化。而我们有时候希望从表达式的类型推断出要定义的变量类型，却不想用该表达式的值初始化变量。</strong>为了满足这一要求，C++11 新标准引入了第二种类型说明符 `decltype`，它的作用是选择并返回操作数的数据类型。

```cpp
int ci = 0;
decltype(ci) z;             // 声明 z，z 的类型就是 ci 的类型 (int)
```

第三章 字符串、向量和数组
===========
本章将介绍两种最重要的标准库类型：`string` 和 `vector`。`string` 表示可变长的字符串序列，`vector` 存放的是某种给定类型对象的可变长序列。在开始它们之前，我们先来学习一种访问库中名字的简单方法。

命名空间的 using 声明
------------
我们通常用 `std::cin` 表示从标准输入中读取内容，这种方法会比较繁琐。其实我们可以使用**using 声明**来直接访问命名空间中的名字：

```cpp
#include <iostream>
using std::cin;

int main(){
    int i;
    cin >> i;
    std::cout << i << std::endl;
    return 0;
}
```

> 一般来说，位于头文件中的声明应该尽量避免使用 `using` 声明。这是因为头文件中的内容会拷贝到所有引用它的文件中去，可能会发生名字冲突，这点需要注意。

标准库类型 string
------------
作为标准库的一部分，`string` 定义在命名空间 `std` 中。接下来的示例都假定已经包含了下述代码：

```cpp
#include <string>
using std::string;
```

### 定义和初始化 string 对象
下面列出了初始化 `string` 对象最常使用的一些方式:

```cpp
string s1;              // 默认初始化，s1 是一个空字符串
string s2 = s1;         // s2 是 s1 的副本
string s3 = "hiya"；    // s3 是该字符串字面的副本
string s4(10, 'c');     // s4 的内容是 cccccccccc
```

C++ 语言有几种不同的初始化方式，通过 `string` 我们可以清楚地看到这些初始化方式之间有什么区别和联系。如果使用等号 `(=)` 初始化一个变量，实际上执行的是**拷贝初始化**，编译器把等号右侧的初始值拷贝到新创建的对象中去。与之相反，如果不使用等号，则执行的是**直接初始化**。 

```cpp
#include <string>
#include <iostream>

int main(){
    std::string s0 = "hello";           // 拷贝初始化
    std::string s1("hello");            // 直接初始化
    std::string s2 = s1;                // 拷贝初始化

    s1[0] = 'n';

    std::cout << s0 << std::endl;       // hello
    std::cout << s1 << std::endl;       // nello
    std::cout << s2 << std::endl;       // hello
    return 0;
}
```

### string 对象上的操作
- `empty` 和 `size` 操作

顾名思义，`empty` 函数根据 `string` 对象是否为空返回一个对应的布尔值，`size` 函数返回 `string` 对象的长度（即 `string` 对象中字符的个数）。对于 `size` 函数来说，它返回的是一个 `string::size_type` 类型值，它是一个无符号的值而且足够存放任何 `string` 的大小。尽管我们不太清楚 `string::size_type` 类型的细节，但是 `C++11` 新标准中允许编译器通过 `auto` 或 `decltype` 来推断变量的类型:

```cpp
auto len = line.size();        // len 的类型是 string::size_type
```

- 两个 `string` 对象相加

两个 `string` 对象相加得到一个新的 `string` 对象，内容是把左侧的运算对象与右侧对象串接而成。

```cpp
string s1 = "hello, ";
string s2 = "world\n";
string s3 = s1 + s2;     // s3 的内容是 hello, world\n
s1 += s2;                // 等价于 s1 = s1 + s2
```
- 字面值和 `string` 对象相加

当 `string` 对象和字符字面值及字符串字面值混合在一条语句中使用时，必须确保每个加法运算符 `(+)` 的两侧的运算对象至少有一个是 `string`。

```cpp
string s4 = s1 + ", ";             // 正确
string s5 = "hello" + ", ";        // 错误
string s6 = s1 + ", " + "world";   // 正确, s1 + ", " 返回的是一个 string 对象
string s7 = "hello" + ", " + s2;   // 错误
```
标准库类型 vector
------------
要想使用 `vector`，必须包含适当的头文件：

```cpp
#include <vector>
```
**vector 是一个类模版**，我们将在模版名字后面一对尖括号，在括号内存放对象的类型：

```cpp
std::vector<int> ivec;                  // ivec 保存 int 类型的对象
std::vector<std::vector<int>> file;  // file 保存的是 vector 对象
```
### 定义和初始化 vector 对象
和任何一种类类型一样，`vector` 模版控制着定义和初始化向量的方法，下表列出了定义 `vector` 对象的常用方法。

```cpp
std::vector<T> v1;                     // v1 是一个空 vector，而且 capacity 也返回 0，意味着还没有分配内存空间
std::vector<T> v2(v1);                 // v2 中包含 v1 所有元素的副本
std::vector<T> v2 = v1;                // 等价于 v2(v1)，v2 中包含有 v1 所有元素的副本
std::vector<T> v3(n, val);             // v3 包含了 n 个重复元素，每个元素的值都是 val
std::vector<T> v4(n);                  // v4 包含了 n 个重复地执行了值初始化的对象
std::vector<T> v5{a,b,c, ...}          // v5 包含了初始值个数的元素，每个元素被赋予相应的初始值
std::vector<T> v5={a,b,c ...}          // 相当于 v5{a,b,c, ...}
```

另外，`.data()` 函数提供了一个能直接指向内存中存储 `vector` 元素位置的指针。由于 `vector` 里面的元素都是顺序连续存放的，该指针可以通过偏移量来访问数组内的所有元素。

- 程序：

```cpp
#include <iostream>
#include <vector>

int main()
{
    std::vector<int> vi = {1,20,30};
    std::cout<< "vi.capacity=" << vi.capacity() <<std::endl;
    
    auto p = vi.data();
    std::cout << "p=" << p << std::endl;
    std::cout << "*p=" <<*p << std::endl;
    std::cout<<"*(p+2)=" << *(p+2)<< std::endl;
}
```

- 输出：

```bashrc
╭─yang@yangdeMacBook-Pro.local ~  
╰─➤  ./a.out 
vi.capacity=3
p=0x7ff2d2405790
*p=1
*(p+2)=30
```


### 向 vector 对象添加元素
向 `vector` 对象添加元素可以使用成员函数 `push_back` 进行添加，它负责把一个元素 “压” (**push**) 到 `vector` 对象到末端。例如：

```cpp
std::vector<int> v2;
for(int i=0; i<100; i++){
    v2.push_back(i);            // 依次把整数放到 v2 尾端
}
```

但是我们其实还有一种更快的添加元素函数，它是 `C++11` 引入的新特性: `emplace_back` 方法。

```cpp
std::vector<int> v2;
for(int i=0; i<100; i++){
    v2.emplace_back(i);         // 依次把整数放到 v2 尾端
}
```

`C++11` 引入了右值引用（转移构造函数），`push_back` 会调用构造函数和右值引用，使用 `emplace_back` 替代 `push_back` 可以在这上面有进一步优化空间，它只调用构造函数不需要调用右值引用转移构造函数。如下面这个例子所示：

```cpp
#include <iostream>
#include <vector>

class A{
    public:
        A (int x_arg) : x (x_arg) { std::cout << "A (x_arg)\n"; }           // 构造函数
        A () { x = 0; std::cout << "A ()\n"; }
        A (const A &rhs) noexcept { x = rhs.x; std::cout << "A (A &)\n"; }
        A (A &&rhs) noexcept { x = rhs.x; std::cout << "A (A &&)\n"; }      // 右值引用

    private:
        int x;
};

int main(){
    std::vector<A> a;
    std::cout << "call push_back:\n";
    a.push_back(0);
    // (1) create temp object and 
    // (2) then move copy to vector and 
    // (3) free temp object
    
    std::vector<A> b;
    std::cout << "call emplace_back:\n";
    b.emplace_back(1);
    // (1) direct object creation inside vector
    return 0;
}
```
输出：

```bashrc
╭─yangyun@yangyundeMBP.lan ~ ‹system› 
╰─➤  g++ test.cpp -std=c++11
╭─yangyun@yangyundeMBP.lan ~ ‹system› 
╰─➤  ./a.out 
call push_back:              # push_back 调用构造函数和右值引用
A (x_arg)
A (A &&)
call emplace_back:           # emplace_back 只调用了构造函数
A (x_arg)
```

> 右值引用：以前我们对一个临时变量进行搬运的时候，通常调用的是拷贝构造函数，它需要重新创建一个新指针并指向一块新内存，然后将原来内存的内容拷贝过来。而现在右值引用是不会分配一块新的内存，而只是创建一个新的指针并指向原来那块内存，原指针就会被销毁掉。

### 向 vector 对象删除元素
假设我们想针对容器 `std::vector<int> vec = {0,1,2,3,4,5,6,7}` 删除掉大于 5 的数字，那么我们可以使用 **iterator** 进行遍历删除。

```cpp
auto iter = vec.begin();
while(iter != vec.end()){
    if(*iter>5){           // 如果值大于 5 则删除
        vec.erase(iter);
    } else{
        iter++;            // 否则继续遍历
    }
}

for(int i=0; i<vec.size(); i++){
    std::cout << vec[i] << " ";             // 0 1 2 3 4 5
}
```
但是我们切记不能通过通过以下 ` i<vec.size()` 的方法去遍历删除，这是因为容器 `vec` 在遍历删除的过程中，容器内的元素是变化的。

```cpp
for(int i=0; i<vec.size(); i++){
    if(vec[i]>5)
        vec.erase(vec.begin()+i);            // vec 删除元素 6 的时候便停止遍历了
}

for(int i=0; i<vec.size(); i++){
    std::cout << vec[i] << " ";             // 0 1 2 3 4 5 7
}

```
### vector 对象的越界检查
`STL` 实现者在对 `vector` 进行内存分配的时候，其实际的容量要比当前所需的空间要多一些。就是说，`vector` 容器<strong>预留了一些额外的存储区，用于存放新添加的元素。</strong>比如 `vec` 虽然只初始化了 8 个元素，但是它可能实际分配了 8000 个元素所需的空间。

- `operator []` 是不会进行越界检查的，使用它可能会发生意想不到的访问错误，但是它的访问速度非常快；
- 使用成员函数 `at` 也可以进行访问，但是每次访问前都会对下标索引进行检查，这使得速度慢了两三倍。

迭代器介绍
------------
对于容器 `vector` 或 `string` 对象，我们可以通过下标索引去直接访问元素。但是对于一些链表结构，我们只能通过节点中指向下一节点的指针去遍历访问，直到我们找到想要的元素为止。

- 数组的 `find` 函数

```cpp
double *find_ar(double *ar, int n, const double &val){
    for(int i=0; i<n; i++){
        if(arr[i] == val)
            return &ar[i];
    return nullptr;
}
```
- 链表的 `find` 函数

```cpp
struct Node{
    double item;
    Node *p_next;
}

Node *find_ll(Node *head, const double &val){
    Node *start;
    for(start == head; start != 0; start = start->p_next)
        if(start->item == val)
            return start;
    return nullptr;
}
```

如果我们有一个迭代器能够直接通过 `p` 访问容器中某一元素，并且通过 `p++` 或 `++p` 去访问下一个元素就好了。（这里 `p` 相当于一个指针）这就是迭代器出现的原因之一。

### 使用迭代器
迭代器里经常使用的是 `begin` 和 `end` 方法：其中 `begin` 成员负责返回指向第一个元素的迭代器，`end` 成员负责返回指向容器 **“尾元素的下一位置（on past the end)”**，也就是说，该迭代器指示的是容器的一个本不存在的 **“尾后”** 元素。这样的迭代器根本没有什么实际含义，仅是个标记而已，表示我们已经处理完了容器中所有元素。

```cpp
for(auto iter=vec.begin(); iter!=vec.end(); iter++)
    if(*iter == val)
        return iter;
return nullptr;
```
### 迭代器运算
`string` 和 `vector` 的迭代器提供了更多额外的运算符，一方面可使得迭代器的每次移动可以跨过多个元素，另外也支持迭代器进行关系运算。

|运算|含义|
|---|---|
|`iter + n`| 迭代器向后移动了 n 个元素的位置 |
|`iter - n`| 迭代器向前移动了 n 个元素的位置 |
|`iter += n`| 迭代器的加法复合赋值语句，将 `iter` 加 n 的值赋给 `iter`|
|`iter -= n`| 迭代器的减法复合赋值语句，将 `iter` 减 n 的值赋给 `iter`|
|`>、>=、 <、<=`| 迭代器的关系运算符，如果某迭代器指向容器位置在另一个迭代器所指位置之前，则说明前者小于后者。|

使用迭代器运算的一个经典算法是二分搜索，二分搜索从有序序列中寻找某个给定的值。二分搜索从序列中间的位置开始搜索，如果中间位置元素正好就是要找的元素，搜索完成；如果不是，假如该元素小于要找的元素，则在序列的后半部分继续搜索；假如该元素大于要找的元素，则在序列的前半部分继续搜索。在缩小范围中计算一个新的中间元素并重复之前的过程，直至最终找到目标或没有元素可以继续搜索。

下面的程序使用迭代器完成了二分搜索。

```cpp
// text 必须是有序的，beg 和 end 表示我们的搜索范围
auto beg = text.begin(), end = text.end();
auto mid = text.begin() + (end - beg) / 2;   // 初始状态的中间位置

while(mid != end && *mid != sought){
    if(sought < *mid)
        end = mid;
    else
        beg = mid + 1;                     // 在 mid 之后寻找
    mid = beg + (end - beg) / 2;           // 新的中间点
}
```

数组
------------
 与 `vector` 相似的地方是，数组也是存放类型相同的对象的容器，这些对象本身没有名字，需要通过其所在的位置访问。与 `vector` 不同的地方是，数组的大小确定不变，不能随意向数组中增加元素。
### 定义和初始化内置数组
数组是一种复合类型，数组的声明形如 `a[d]`，其中 `a` 是数组的名字，`d` 是数组的维度。维度说明了数组中元素的个数，因此必须大于 0。一般而言，初始化数组通常会遇到有以下几种方式：

```cpp
int a[] = {1,2,3};     // a 含有 3 个整数元素
int *b[10];            // b 含有 10 个整数型指针的数组
int (*c)[10];          // c 指向一个含有 10 个整数的数组
ing (&d)[10];          // d 引用一个含有 10 个整数的数组
```
### 访问数组元素
与标准库类型 `vector` 和 `string` 一样，数组的元素也能使用 `for` 语句或下标运算符来访问。在使用数组下标的时候，通常将其定义为 `size_t` 类型。**`size_t` 是一种机器相关的无符号类型，它被设计得足够大以便能表示内存中任意对象的大小**。

```cpp
int array[10];
for(size_t i=0; i<10; i++)
    array[i] = 1;
```
### 指针和数组
在 `C++` 语言中，指针和数组有非常紧密的联系。就如即将介绍的，使用数组的时候编译器一般会将它转换成指针。数组的元素也是对象，因此向其他对象一样，对数组元素取地址符就能得到指向该元素的指针。

```cpp
string nums[] = {"one", "two", "three"};
string *p1 = &nums[2];                     // p1 指向 nums 的第三个元素
```

然而，数组还有一个特性：在很多用到数组名字的地方，编译器会自动地替换为一个指向数组首元素的指针：`string *p2 = nums;` 它等价于 `string *p2 = &nums[0]`。

就像使用迭代器遍历 `vector` 对象中的元素一样，使用指针也能遍历数组中的元素。之前介绍过，通过数组的名字或者数组中首元素的地址都能得到指向首元素的指针。给指针加上一个整数，得到的新指针仍指向同一数组的其他元素。

```cpp
int arr[] = {1,2,3,4,5};
int *ip1 = arr;             // 等价于 int *ip1 = &arr[0]
int *ip2 = ip1 + 2;         // ip2 指向 arr 的第三个元素 arr[2]
```

只要指针指向的是数组的元素，都可以执行下标运算：

```cpp
int j = ip2[1];             // 等价于 j = arr[3]
int k = ip2[-2];            // 等价于 k = arr[0]
```
第四章 表达式
===========

表达式是由一个或多个运算对象组成，对表达式求值将得到一个结果。字面值和变量是最简单的表达式，其结果就是字面值和变量的值。把一个运算符和一个或多个运算对象组合起来可以生成较为复杂的表达式。

左值和右值
------------
`C++` 的表达式要不然是**右值（rvalue, 读作 “are-value”）**，要不然就是**左值（lvalue, 读作 “ell-value”）**。可以对它们作一个简单的归纳：当一个对象被用作右值的时候，用的是对象的值（内容）；当对象被用作左值的时候，用的是对象的身份（在内存中的位置）。

### 通俗的说法

还有另一种更容易理解的说法：**在C++11中可以取地址的、有名字的就是左值，反之，不能取地址的、没有名字的就是右值**。例如，`int a = b+c`, `a` 就是左值，其有变量名为 `a`，通过 `&a` 可以获取该变量的地址；而表达式 `b+c`，我们不能通过变量名找到它，`＆(b+c)` 这样的操作则不会通过编译。

### 左值引用
左值引用通常也不能绑定到右值，而是绑定在已初始化的左值上：

```cpp
int &a = 2;       # 左值引用绑定到右值，编译失败
int b = 2;        # 非常量左值
const int &c = b; # 常量左值引用绑定到非常量左值，编译通过
```

### 右值引用
右值值引用通常不能绑定到任何的左值，要想绑定一个左值到右值引用，通常需要 `std::move()` 将左值强制转换为右值，例如：

```cpp
int a = 1;
int &&r1 = a;             # 编译失败
int &&r2 = std::move(a);  # 编译通过
```

> 右值引用和左值引用都是属于引用类型。无论是声明一个左值引用还是右值引用，都必须立即进行初始化。而其原因可以理解为是引用类型本身自己并不拥有所绑定对象的内存，只是该对象的一个别名。

例如，C++11 给 `string` 类添加了移动语义，这意味着添加一个**移动构造函数**时，它使用右值引用而非左值引用：

```cpp
basic_string(basic_string &&str) noexcept;
```

在实参为临时对象时将调用这个构造函数：

```cpp
string one("din");         // C-style string constructor
string two(one);           // copy constructor - one is an lvalue
string three(one+two);     // move constructor - sum is an rvalue
```

在上面中，`three` 将获取 `operator+()` 创建的对象的所有权，**而不是将对象复制给 `three`，然后再销毁对象**。


各种运算符
------------

### 逻辑运算符
|运算符|功能|用法|
|---|---|---|
|`&&`|逻辑与|`expr && expr`|
|`!`|逻辑非|`!expr`|

### 递增和递减运算符
递增和递减运算符有两种形式：前置版本和后置版本，它们的的区别在于返回值的不同。

```cpp
int i = j = 0, k;
k = ++i;    // k = 1, i = 1
k = j++;    // k = 0, j = 1
```

举个例子，可以使用这种递增运算符来控制循环输出一个 `vector` 对象内容直至遇到（但不包括）第一个负值为止:

```cpp
auto pbeg = v.begin();
// 输出元素直至遇到第一个负值为止
while (pbeg != v.end() && *pbeg >= 0)
    std::cout << *pbeg++ << std::endl; // 输出当前值并将 pbeg 向前移动一个元素
```

形如 `*pbeg++` 的表达式一开始可能不太容易理解，但其实这是一种被广泛使用的、有效的写法。当对这种形式熟悉之后，书写

```cpp
std::cout << *iter++ << std::endl;
```

要比书写下面的等价语句更简洁、也更少出错

```cpp
std::cout << *iter << std::endl;
++iter;
```

### 条件运算符
条件运算符 `?:` 允许我们把简单的 `if-else` 逻辑潜入到单个表达式当中，条件运算符按照如下形式使用：

```cpp
cond? expr1: expr2;
```
它执行的过程是：首先求 `cond` 的值，如果条件为真对 `expr1` 求值并返回该值，否则对 `expr2` 求值并返回该值。举个例子，我们可以使用条件运算符判断成绩是否合格：

```cpp
string final_grade1 = (grade < 60)? "fail": "pass";
string final_grade2 = (grade > 90)? "high pass"
                                  : (grade < 60) ? "fail": "pass";
```

### 移位运算符
移位运算是对其运算对象执行基于二进制位的移动操作，首先令左侧运算对象的内容按照右侧运算对象的要求移动指定位数，然后将经过移动的左侧运算对象的拷贝作为求值结果。二进制位或者向左移`（<<)` 或者向右移`（>>）`，移出边界之外的位就被舍弃掉了。

```cpp
unsigned char bits = 0233;    
```
`unsigned char` 保存只有一个字节，即 8 位。但是它首先会将 `bits` 提升成 `int` 类型（32位），因此`0233` 的二进制表示如下：

`0` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0` | `1` `0` `0` `1` `1` `0` `1` `1`

- 向左移动 8 位

`0` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0` | `1` `0` `0` `1` `1` `0` `1` `1` | `0` `0` `0` `0` `0` `0` `0` `0`

- 向左移动 31 位，左边超出边界的位丢弃掉了

`1` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0`

- 向右移动 3 位，最右边的 3 位舍弃掉了

`1` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `0` `0` `0` `0` `0` | `0` `0` `0` `1` `0` `0` `1` `1`

类型转换
------------
举个例子，考虑下面这条表达式，它的目的是是将 `ival` 初始化为 `6`。

```cpp
int ival = 3.543 + 3;                         // 编译器可能会警告该运算损失了精读
```

加法的两个运算对象类型不同：3.543 的类型是 `double`，3 的类型是 `int`。C++ 语言不会将两个不同类型的对象相加，而是先根据类型转换规则设法将运算对象的类型统一后再求值。上述的类型是自动执行的，无须程序员的介入，它们被称为**隐式转换**。有时我们希望显示转换对象强制转换成另外一种类型，C++ 给我们提供了 4 种显示转换的方式。

### static_cast
任何具有明确定义的类型转换，只要不包含底层 `const` ，都可以使用 `static_cast`。例如，通过将一个运算对象强制转换成 `double` 类型就能使表达式执行浮点数除法。

```cpp
int i = j = 2;
double slope = static_cast<double>(j) / i;       // 编译器将不会发出警告信息
```

`static_cast` 对于编译器无法自动执行的类型转换也非常有用。例如，我们可以使用 `static_cast` 找回存在于 `void*` 指针中的值。

```cpp
void *p = &d;                              // 正确：任何非常量对象的地址都可以存入 void*
double *dp = static_cast<double*>(p);      // 正确：将 void* 指针转换回初始指针的类型
```
### dynamic_cast
`dynamic_cast` 是运行阶段类型识别（RunTime Type Identification，简称**RTTI**) 的一种，它不能回答“指针指向的是哪类对象”这样的问题，但能回答“是否可以安全地将对象的地址赋给特定类型的指针”这样的问题。先来看看下面一段程序：

程序：

```cpp
#include <iostream>

class Base{
    public:
        virtual void foo(){
            std::cout << "调用基类方法" << std::endl;
        }
};

class Derived: public Base{
    public:
        virtual void foo(){
            std::cout << "调用子类方法" << std::endl;
        }
};

// dynamic_cast 在运行时会进行检查，开销比 static_cast（没有开销）大一些

int main(){
    Base*    pg = new Base;
    Derived* ps = new Derived;

    // 不能将基类指针转化成派生类指针，因为派生类的方法基类不一定有，因此转化失败，得到 NULL 指针
    Derived* pa = dynamic_cast<Derived*>(pg);
    if(pa == NULL){
        std::cout << "pa == NULL" << std::endl;
    }
    
    // 下面比较 dynamic_cast 和 static_cast 两种转换方式:

    Base* pb = dynamic_cast<Base*>(ps);
    if(pb != NULL){
        std::cout << pb << std::endl;
        pb->foo();                            // pb 虽然是基类指针，但是会根据 RTTI 原则调用子类的方法
    }

    Derived* pc = static_cast<Derived*>(pg);  // 在编译器进行转化，不会在运行期进行检查，访问的依然是 Base 的 foo 方法
    if(pc != NULL){
        std::cout << pc << std::endl;
        pc->foo();
    }

    return  0;
}
```

输出：

```bashrc
╭─yangyun@yangyundeMBP.lan ~ ‹system› 
╰─➤  ./a.out 
pa == NULL
0x7faaf0d02720
调用子类方法
0x7faaf0d02710
调用基类方法
```

这里提出了这样的问题：基类指针 `pg` 的类型是否可以被安全地转换为 `Derived*`？如果可以，运算符将返回对象的地址，否则返回一个空指针。显然 `dynamic_cast` 这里对这个转换做了检查，认为这种转换非法，因此返回了一个 `NULL` 指针。

> 将基类对象的地址赋给子类指针是非法的，这是因为子类可能会包含一些基类没有的数据成员和方法，然而将子类对象的地址赋值给基类指针是安全的。

### const_cast
提供 `const_cast` 运算符的原因是，有时候可能需要这样一个值，它在大多数的时候是常量，而有时又是可以修改的。在这种情况下，可以将这个值声明为 `const`，并在需要修改它的时候，使用 `const_cast`。

程序：

```cpp
#include <iostream>

int main(){
    const int const_val = 26;

    int* ptr = const_cast<int*>(&const_val);   // 将 const_val 的地址通过 const_cast 运算符赋值给指针 ptr
    *ptr = 3;                                  // 指针 ptr 可以修改它指向的值

    std::cout<< "constant:  " << const_val << std::endl;
    std::cout<< "    *ptr:  " << *ptr      << std::endl;
}
```

输出：

```bashrc
╭─yangyun@yangyundeMBP.lan ~ ‹system› 
╰─➤  ./a.out 
constant:  26
    *ptr:  3
```

非常趣味的是：`C++` 还是很厚道的，对声明为 `const`的变量 `const_val` 来说，常量就是常量，任你各种转化，常量的值就是不会变，这是C++的一个承诺。
### reinterpret_cast
`reinterpret_cast` 通常为运算对象的位模式提供较低层次上的重新解释, 其作用为: **允许将任何指针转换为任何其他指针类型。 也允许将任何整数类型转换为任何指针类型以及反向转换**。举个例子，假设有如下的转换。

程序:

```cpp
#include <iostream>

int main(){
    int a = 65;
    char *p1 = reinterpret_cast<char*>(&a);
    std::cout << *p1 << std::endl;
    return 0;
}
```

输出:

```bashrc
╭─yangyun@yangyundeMBP.lan ~ ‹system› 
╰─➤  ./a.out 
A
```

上述转化是可以的，因为它其实是将一个 `int` 类型的变量转化为 `char` 变量（`A` 的 `ASCII` 码为 `65`）。接下来再使用 `p1` 时就会认定它的值是 `char*` 类型，编译器没法知道它实际存放的是指向 `int` 的指针。这可能会引发一些严重的后果，而查找这类问题的原因会变得非常困难。

```cpp
int b = 0x00636261;
char *p2 = reinterpret_cast<char*>(b);    // 直接将整数 b 强行解释为 char 类型的地址
```

如果我们想要访问指针 p2 指向的值 ```std::cout << *p2 << std::endl;``` 就会出现段错误(`segmentation fault`)。因为这块地址压根不存在，属于非法地址。

> reinterpret_cast 本质上非常依赖于机器，要想安全使用它，就必须对涉及的类型和编译器实现转换的过程非常了解，建议谨慎使用。


第五章 语句
===========
简单语句和作用域
------------
C++ 语言中的大多数语句都以分号结束，一个表达式，比如 `ival + 5`，末尾加上分号就变成了表达式语句。表达式语句的作用是执行表达式并丢弃掉求值结果:

```cpp
ival + 5;                            // 一条没什么实际用处的表达式语句
std::cout << ival;                   // 一条有用的语句
```

可以在 `if, switch, while` 和 `for` 语句的控制结构内定义变量。定义在控制结构当中的变量只在相应语句的内部可见，一旦语句结束，变量也就超出其作用范围了。

```cpp
while (int i = get_num())
    std::cout << i << std::endl;         // 每次迭代时创建并初始化 i
i = 0;           // 错误：在循环外部无法访问 i
```

条件语句
------------
C++ 语言提供了两种按条件执行的语句。一种是 `if` 语句，它根据条件决定控制流；另外一种是 `switch` 语句，它计算一个整型表达式的值，然后根据这个值从几条执行路径中选一条。
### if 语句
`if` 语句的作用是：判断一个指定的条件是否为真，根据判断的结果决定是否执行另外一条语句。

```cpp
#include <vector>
#include <iostream>
#include <string>

int main(){
    int grade = 80;

    std::string lettergrade;
    const std::vector<std::string> scores = {"F", "D", "C", "B", "A", "A++"};

    if(grade < 60)
        lettergrade += "+";
    else{
        lettergrade = scores[(grade-50)/10];
        if(grade % 10 > 7)
            lettergrade += "+";
        else if(grade % 10 < 3)
            lettergrade += "-";
    }
    return 0;
}
```
### switch 语句
`switch` 语句提供了一条便利的途径使得我们能够在若干固定选项中做出选择。举个例子，假如我们想统计 5 个元音字母和其他字母在文本中出现的次数，程序逻辑应该如下所示：

程序：

```cpp
#include <vector>
#include <iostream>
#include <string>

int main(){
    std::string str;
    std::cin >> str;

    unsigned aCnt = 0, eCnt = 0, iCnt = 0, oCnt = 0, uCnt = 0, otherCnt = 0;
    for(int i=0; i<str.size(); i++){
        char ch = str[i];
        switch(ch){
            case 'a':
                ++aCnt;
                break;
            case 'e':
                ++eCnt;
                break;
            case 'i':
                ++iCnt;
                break;
            case 'o':
                ++oCnt;
                break;
            case 'u':
                ++uCnt;
                break;
            default:             // 如果前面都不是，那就属于其他字母
                ++otherCnt;
                break;
        }
    }
    std::cout << "Number of vowel a: \t" << aCnt << '\n'
              << "Number of vowel e: \t" << eCnt << '\n'
              << "Number of vowel i: \t" << iCnt << '\n'
              << "Number of vowel o: \t" << oCnt << '\n'
              << "Number of vowel u: \t" << uCnt << std::endl;
    return 0;
}
```
输出:

```bashrc
╭─yangyun@yangyundeMBP.lan ~ ‹system› 
╰─➤  ./a.out 
hello world
Number of vowel a: 	0
Number of vowel e: 	1
Number of vowel i: 	0
Number of vowel o: 	1
Number of vowel u: 	0
```

迭代语句
------------
迭代语句通常称为循环，它重复执行操作直到满足某个条件才停下来。`while` 和 `for` 语句在执行循环体之前检查条件，`do while` 语句先执行循环体，然后再检查条件。

### while 语句
只要条件为真，`while 语句` 就会重复地执行循环体，它的语法形式是:

```
while (condition)
    statement
```

在 `while` 结构中，只要 `condition` 的求值结果为真就一直执行 `statement`。`condition` 不能为空，如果 `condition` 第一次求值就得 `false`，`statement` 一次都不执行。

### 传统的 for 语句
for 语句的语法形式是:

```
for (init-statement; condition; expression)
    statement
```

一般情况下，`init-statement` 负责初始化一个值，这个值将随着循环的进行二改变，`condition` 作为控制循环的条件，只要 `condition` 为真，就执行一次 `statement`。`expression` 负责修改 `init-statement` 初始化的变量，这个变量刚好是 `condition` 检查的对象。

### 范围 for 语句
C++11 新标准引入了一种更简单的 `for` 语句，这种语句可以遍历容器和其他序列的所有元素。**范围 for 语句**的语法形式是:

```
for (declaration: expression)
    statement
```

`expression` 表示的必须是一个序列，比如用花括号括起来的初始值列表、数组或者 `vector` 或 `string` 等类型的对象，这些类型的共同特点是拥有能返回迭代器的 `begin` 和 `end` 成员。

```cpp
std::vector<int> v = {1,2,3,4,5,6,7,8};
for(auto &r: v)
    r *= 2;             // 将 v 中每个元素的值翻倍
```

`for` 语句头声明了循环控制变量 `r`，并把它和 `v` 关联在一起，我们使用 `auto` 令编译器为 `r` 指定正确的类型。**由于准备修改 `v` 元素的值，因此将 `r` 声明成引用类型**。此时，在循环内给 `r` 赋值，即改变了 `r` 所绑定元素的值。

### do while 语句
`do while 语句` 和 `while` 语句非常相似，唯一的区别是：`do while` 语句先执行循环体后检查条件。**不管条件的值如何，我们都至少执行一次循环**。`do while` 语句的语法形式如下所示:

```
do
    statement
while (condition)
```

在 `do` 语句中，求 `condition` 的值之前首先执行一次 `statement`，`condition` 不能为空。如果 `condition` 的值为假，循环终止；否则，重复循环过程。

```cpp
#include <stdio.h>
 
int main (){
   int a = 10;              /* 局部变量定义 */
   
   /* do 循环执行，在条件被测试之前至少执行一次 */
   do{
       printf("a 的值： %d\n", a);
       a = a + 1;
   }while( a < 20 );
   
   return 0;
}
```

try 语句块和异常处理
------------
程序的异常检测部分使用 `throw` 表达式引发一个异常，然后使用 `try-catch` 语句去捕获异常。先做一个实验：以一个计算两个数的调和平均数的函数为例。两个数的调和平均数的定义是：这两个数字倒数的平均值的倒数，因此程序可以写为：

```cpp
#include <iostream>

double hmean(double a, double b){
    if(a == -b){
        throw "bad hmean() arguments: a = -b not allowed";
    }
    return 2.0 * a * b / (a + b);
}

int main(){
    double x, y, z;

    std::cout << "Enter two numbers: ";
    while(std::cin >> x >> y){
        try{
            z = hmean(x, y);
        }
        catch(const char* s){
            std::cout << s << std::endl;
            std::cout << "Enter a new pair of numbers: ";
            continue;
        }
        std::cout << "Harmonic mean of " << x << " and " << y
                  << " is " << z << std::endl;
        std::cout << "Eneter next set of numbers <q to quit>: ";
    }
    std::cout << "Bye!\n";
    return 0;
}
```

当 `a==-b` 时，引发异常的是字符串 “bad hmean() arguments: a = -b not allowed”。程序会将它赋给变量 `s`，然后执行处理程序中的代码。

第六章 函数
===========
参数传递
------------
函数每次被调用的时候都会被创建它的**形参**，并用传入的**实参**对形参进行初始化。和其他变量一样，形参的类型决定了形参和实参交互的方式。如果形参是引用类型，它将绑定到对应的实参上；否则，将实参数的值拷贝后赋给形参。

> 形参出现在函数定义中，在整个函数体内都可以使用， 离开该函数则不能使用。实参出现在主调函数中，进入被调函数后，实参变量也不能使用。形参和实参的功能是作数据传送。发生函数调用时， 主调函数把实参的值传送给被调函数的形参从而实现主调函数向被调函数的数据传送。

### 传值参数
当初始化一个非引用类型的变量时，初始值被拷贝给变量。此时，对变量的改动不会影响初始的值。例如某个函数试图根据传值行为去改变某个指针，这是徒劳的。

```cpp
void reset(int *ip){
    *ip = 0;      // 改变指针 ip 所指对象的值，这个是可以的；
    ip = 0;        // 只改变了 ip 的局部拷贝，实参未被改变
}
```

调用 `reset` 函数之后，实参所指的对象被指为 0， 但是实参本身并未改变：

```cpp
int i = 42;
reset(&i);                                              // 只能改变 i 的值而非 i 的地址
std::cout << "i = " << i << std::endl;         // 输出 i=0 
```

### 传引用参数
回忆过去所学的知识，我们知道对引用的操作实际上是作用在引用所引的对象上。引用传参的行为与之类似，通过使用引用传参，允许函数改变一个或多个实参的值。举个例子，我们可以改写上一个小节的 `reset` 程序，使其接受的参数是引用类型而非指针：

```cpp
void reset(int &i){
    i = 0;                                // 改变了 i 所引对象的值
}
```

**拷贝大的类类型对象或者容器对象比较低效**，甚至有的类类型（包括 `IO` 类型在内）根本就不支持拷贝操作。当某种类型不支持拷贝操作时，函数只能通过引用形参访问该类型的对象。

### 数组形参
数组的两个特殊性质对我们定义和使用在数组上的函数有影响，这两个性质分别是：不允许拷贝数组以及使用数组时通常会将其转换成指针。因为不能拷贝数组，所以我们无法以值传递的方式使用数组参数。因为数组会被转换成指针，所以**当我们为函数传递一个数组时，实际上传递的是指向数组首元素的指针**。

尽管不能以值传递的方式传递数组，但是我们可以把形参写出类似数组的形式：

```cpp
void print(const int*);
void print(const int []);                 // 可以看出来，函数的意图是作用于一个数组
void print(const int [10]);               // 这里的维度是我们期望数组含有多少个元素，实际不一定
```

尽管表现方式不同，但上面的三个函数是等价的：每个函数的唯一形参都是 `const int*` 类型的。当编译器处理对 `print` 函数的调用时，只检查传入的参数是否是 `const int*` 类型:

```cpp
int i=0, j[2] = {0, 1};
print(&i);                              // 正确：&i 的类型是 int*
print(j);                                // 正确：j 转换成 int* 并指向 j[0]
```

> 如果我们传给 `print` 函数的是一个数组，则实参自动地转换成指向数组首元素的指针，数组的大小对函数的调用没有影响。和其他使用数组的代码一样，以数组作为形参的函数也必须确保使用数组时不会越界。

我们曾经介绍过，在 `C++` 语言中实际上没有真正的多维数组，所谓的多维数组其实是数组的数组。和所有数组一样，当将多维数组传递给函数时，真正传递的是指向数组首元素指针。因为我们处理的是数组的数组，所以首元素本身就是一个数组，指针就是一个指向数组的指针。数组的第二维大小都是数组类型的一部分，不能忽略：

```cpp
void print(int (*matrix)[10], int row_size){ /* ... */};
```
等价定义

```cpp
void print(int matrix[][10], int row_size){ /* ... */};
```

再一次强调，`*matrix` 两端的括号必不可少：

```cpp
int *matrix[10];              // 10 指针构成的数组
int (*matrix)[10];            // 指向含有 10 个整数的数组的指针
```

返回类型和 return 语句
------------
### 无返回值函数
没有返回值的 `return` 语句只能用在返回类型是 `void` 的函数中。返回 `void` 的函数不要求非得有 `return` 语句，因为这类函数的最后一句后面会隐式地执行 `return`。例如，可以编写一个 `swap` 函数，使其在参与交换的值相等时什么也不做就直接退出。

```cpp
void swap(int &v1, int &v2){
    if (v1 == v2)
        return;              // 如果两个值是相等的，则无需交换就直接退出
    int tmp = v2;
    v2 = v1;
    v1 = tmp;
    // 此处无需显式的 return 语句
}
```

### 有返回值函数

- a. 不要返回局部对象的引用或指针
函数完成后，它所占用的存储空间也随之被释放掉。因此函数终止意味着局部变量的引用将指向不再有效的内存区域：

```cpp
#include <string>

const std::string &mainp(){
    std::string ret = "hello";
    if (!ret.empty())
        return ret;              // 错误，返回局部对象的引用
    else
        return "empty";          // 错误，"empty" 是一个局部临时量
}
```
- b. 引用返回左值

函数的返回类型决定函数调用的是否是左值。**调用一个返回引用的函数得到左值，其他返回类型得到右值**。可以像使用其他左值那样来使用返回引用的函数调用，特别是：我们能为返回类型是非常量引用的函数的结果赋值。

```cpp
#include <string>
#include <iostream>

char &getVal(std::string &str, std::string::size_type ix){
    return str[ix];
}

int main(){
    std::string s("hello");
    std::cout << s << std::endl;           // 输出 hello
    getVal(s, 0) = 'A';
    std::cout << s << std::endl;           // 输出 Aello
}
```

- c. 列表初始化返回值

C++11 新标准规定，函数可以返回花括号包围的值的列表。类似于其他返回结果，此处的列表也用来对函数返回的临时量进行初始化。举个例子，在下面的函数中，我们返回一个 `vector` 对象，用它存放表示错误信息的 `string` 对象：

```cpp
vector<string> process(){
    // expected 和 actual 是 string 对象
    if (expected.empty())
        return {};                                         // 返回一个空 vector 对象
    else if (expected == actual)
        return {"functionX", "okay"};                      // 返回列表初始化的 vector 对象
    else
        return {"functionX",  expected, actual};
}
```

- d. 主函数 main 的返回值
之前介绍过，如果函数的返回类型不是 `void`，那么它必须返回一个值。但是这条规定则有个例外：我们允许 `main` 函数没有 `return` 语句直接结束。**如果控制达到了 `main` 函数的结尾处而没有 `return` 语句，编译器将自动隐式地插入一条返回 0 的 `return` 语句**。

函数重载
------------
如果同一个作用域内的几个函数名字相同但形参列表不同，我们称之为**重载（overloaded）函数**。例如，在前面我们定义了几个名为 `print` 的函数：

```cpp
void print(const char *cp);
void print(const int *beg, const int *end);
void print(const int ia[], size_t size);
```

这些函数的名字相同，但是接受的形参类型不一样。当调用这些函数时，编译器会根据传递的实参类型推断想要的是哪个函数：

```cpp
int j[2] = {0, 1};
print("hello world");               // 调用 print(const char *cp)
print(j, end(j)-begin(j));          // 调用 print(const int ia[], size_t size)
print(begin(j), end(j));             // 调用 print(const int *beg, const int *end)
```

内联函数和 constexpr 函数
------------
### 内联函数

假设我们需要编写一个小的函数，它的功能是比较两个 `string` 形参的长度并返回长度较小的 `string` 引用。

```cpp
const string &shorterString(const string &s1, const string &s2){
    return s1.size() <= s2.size() ? s1 : s2;
}
```

如果我们将 `shorterString` 函数指定为**内联函数（`inline`)，通常就是将它在每个调用点上“内联地”展开。例如，以下调用**：

```cpp
cout << shorterString(s1, s2) << endl;
```

将在编译过程中展开类似下面的形式：

```cpp
cout << (s1.size() <= s2.size() ? s1 : s2) << endl;
```

**从而消除了 `shorterString` 函数的调用开销**。我们只要在 `shorterString` 函数的返回类型前面加上关键字 `inline`，这样就可以将它声明成内联函数了。

### constexpr 函数

**`constexpr` 函数** 是指能用于常量表达式的函数。定义 `constexpr` 函数的方法与其他函数类似，不过要遵循几项约定：函数的返回类型及所有的形参类型都得是字面值类型。而且函数体中必须有且只有一条`return`语句：

```cpp
constexpr int new_sz() {return 42;}
constexpr int foo = new_sz();              // 正确：foo 是一个常量表达式
```

我们把 `new_sz` 定义成无参数的 `constexpr` 函数，因为编译器能在程序编译时验证 `new_sz` 函数返回的是常量表达式，所以我们可以直接用 `new_sz` 函数初始化 `constexpr` 类型变量 `foo`。执行该初始化任务时，编译器对 `constexpr` 函数的调用替换成其结果值。**为了能在编译过程中展开，`constexpr` 函数被隐式地指定为内联函数**。

函数指针和回调函数
------------

函数指针指向的是函数而非对象，和其他指针一样，函数指针指向某种特定类型。**函数的类型由它的返回类型和形参类型共同决定，与函数名无关**。例如：

```cpp
void (*pf)(int );
```

该函数的类型是 `void (int)`。要想声明一个可以指向该函数的指针，只需要用指针替换函数名就可以：

```cpp
// pf 指向一个函数，该函数的参数是 int，返回的是 void 类型
void (*fp)(int );    // 未初始化
```

从我们声明的名字开始观察，`pf` 前面有个`*`，因此 `pf` 是指针；右侧是形参列表，表示 `pf` 指向的是函数；再观察左侧，发现函数的返回类型是 `void` 值。因此 `pf` 就是一个指向函数的指针，其函数的参数为 `int` 类型，返回值是 `void` 类型。

> `*pf` 两端的括号必不可少。如果不写这对括号，则 `pf` 是一个返回值为 `void` 指针的函数。

例如，我们可以写这样一个程序：函数 `test` 的形参为一个数组 `array` 和一个函数指针 `*pf`，其功能是通过函数指针 `*pf` 将打印函数 `print` 回调执行打印出数组 `array` 的每一个值。

程序：

```cpp
#include <vector>
#include <iostream>
#include <string>

void print(int number){
    std::cout << number << std::endl;
}

void test(int* array, void (*pf)(int )){
    for (int i = 0; i < 4; i++){
        (*pf)(array[i]);
    }
}

int main(){
    int array[] = {1,2,3,4};
    test(array, &print);       // 由于函数的名字即地址，因此 test(array, print) 也行
}
```

输出：

```bashrc
╭─yang@yangdeMacBook-Pro.local ~  
╰─➤  ./a.out 
1
2
3
4
```

> 由于函数的名字就是它的地址，因此传入 `print` 和 `&print` 给 `test` 函数都是一样的。

在经常使用函数指针之后，我们很快就会发现，每次声明函数指针都要带上长长的形参和返回值，非常不便。这个时候，我们应该想到使用 `typedef`，即为某类型的函数指针起一个别名，使用起来就方便许多了。例如，对于前面提到的函数可以使用下面的方式声明：

```cpp
typedef void (*fp)(int );          // 为该函数指针类型起一个新的名字 fp
fp p;                              // 声明一个指向 fp 类型的函数指针 p
```

如果不想用 `typedef`，也可以用 C++11 的新特性 `using` ，也是一样的效果：

```cpp
using fp = void (*)(int);
```

从而新 `test` 函数的写法就变得很简单了：

```cpp
void test(int* array, fp p){
    for (int i = 0; i < 4; i++){
        (*p)(array[i]);
    }
}
```

第七章 类
===========
类的定义和对象的创建
------------
类是创建对象的模板，一个类可以创建多个对象，每个对象都是类类型的一个变量；创建对象的过程也叫类的实例化。每个对象都是类的一个具体实例（Instance），拥有类的成员变量和成员函数。与结构体一样，类只是一种复杂数据类型的声明，不占用内存空间。而对象是类这种数据类型的一个变量，或者说是通过类这种数据类型创建出来的一份实实在在的数据，所以占用内存空间。

一个简单类的定义：

```cpp
class Student{
    public:
        // 成员变量
        char *name;
        int age;
        float score;

        // 成员函数
        void say(){
            std::cout<< name << "的年龄是"<< age <<"，成绩是" << score << std::endl;
        }
};
```
有了 Student 类后，就可以通过它来创建对象了，例如：

```cpp
Student liLei;        // 创建了一个对象
Student allStu[100];  // 创建了一个 allStu 数组，它拥有 100 个元素，每个元素都是 Student 类型的对象
```

访问控制和封装
------------
到目前为止，我们已经为类定义了借口，但并没有任何访问机制强制用户使用这些接口。我们的类还没有封装，也就是说，用户可以直达 Student 对象的内部并且控制它的实现细节。在 C++ 语言中，我们使用访问说明符加强类的封装性：

- 定义 **public** 说明符之后的成员在整个程序内可被访问，public 成员定义类的接口；
- 定义 **private** 说明符之后的成员可以被类的成员函数访问，但是不能被使用该类的代码访问，private 部分封装了（即隐藏了）类的实现细节。

再一次定义 Student 类，其新形式如下：

```cpp
class Student{
    private:
        // 成员变量
        char *name;
        int age;
        float score;
        
    public:
        // 成员函数
        void say(){
            std::cout<< name << "的年龄是"<< age <<"，成绩是" << score << std::endl;
        }
};
```

在 C++ 中，也可以使用 struct 关键字替代 class 。它们在大多数的情况下是相同的，区别在于：**使用 class 时，类中的成员默认都是 private 属性的；而使用 struct 时，结构体中的成员默认都是 public 属性的。**

### 构造函数和析构函数

每个类都定义了它的对象被初始化的方式，类则通过构造函数（constructor）来控制其对象的初始化过程。构造函数的任务是初始化类对象的数据成员，无论何时只要类的对象被创建，就会执行构造函数。**构造函数的名字和类名字相同。和其他函数不一样的是，构造函数没有返回类型。**

```cpp
class Student{
    private:
        // 成员变量
        char *name_;
        int age_;
        float score_;
        
    public:
        // 成员函数
        Student(char *name, int age, float score):
        	  name_(name), age_(age), score_(score){};
        void say(){
            std::cout<< name_ << "的年龄是"<< age_ <<"，成绩是" << score_ << std::endl;
        }
};
```

创建对象时系统会自动调用构造函数进行初始化工作，同样，销毁对象时系统也会自动调用一个函数来进行清理工作，例如释放分配的内存、关闭打开的文件等，这个函数就是析构函数。

析构函数（Destructor）也是一种特殊的成员函数，没有返回值，不需要程序员显式调用（程序员也没法显式调用），而是在销毁对象时自动执行。构造函数的名字和类名相同，而析构函数的名字是在类名前面加一个 `~` 符号。

```cpp
class Student{
    public:
        // 成员函数
        ~Student(){}:
}
```

> 注意：析构函数没有参数，不能被重载，因此一个类只能有一个析构函数。如果用户没有定义，编译器会自动生成一个默认的析构函数。

### 友元
类可以允许其他类或者函数访问它的非公有成员，方法是令其他类或者函数成为它的**友元（friend)**。如果类想把一个函数作为它的友元，只需要增加一条以 friend 关键字开始的函数声明语句即可：

```cpp
class Student{
    friend void printName(Student &stu);
    ...
};
```

需要注意的是，**友元不是类的成员**。并且友元的声明只能出现在类定义的内部，但是在类内出现的具体位置不限。一般来说，最好在类定义开始或结束前的位置集中声明友元。

### this 指针详解
`this` 是 C++ 中的一个关键字，也是一个 const 指针，它指向当前对象，通过它可以访问当前对象的所有成员。所谓当前对象，是指正在使用的对象。例如对于 `stu.show()`;，`stu` 就是当前对象，`this` 就指向 `stu`。

下面是使用 this 的一个完整示例：

```cpp
class Student{
	...
	
    public:
        void say() const;       // 加 const 表示该函数不会改变成员变量
};

void Student::say() const{
    std::cout<< this->name_ << "的年龄是"<< this->age_ <<"，成绩是" << this->score_ << std::endl;
}
```
this 只能用在类的内部，通过 this 可以访问类的所有成员，包括 private、protected、public 属性的。this 虽然用在类的内部，但是只有在对象被创建以后才会给 this 赋值，并且这个赋值的过程是编译器自动完成的，不需要用户干预，用户也不能显式地给 this 赋值。几点需要注意：

- this 是 const 指针，它的值是不能被修改的，一切企图修改该指针的操作，如赋值、递增、递减等都是不允许的；
- this 只能在成员函数内部使用，用在其他地方没有意义，也是非法的；
- 只有当对象被创建后 this 才有意义，因此不能在 static 成员函数中使用（后续会讲到 static 成员）。

- 1. this 到底是什么?

**this 实际上是成员函数的一个形参**，在调用成员函数时将对象的地址作为实参传递给 this。不过 this 这个形参是隐式的，它并不出现在代码中，而是在编译阶段由编译器默默地将它添加到参数列表中。

- 2. 返回 *this 的成员函数

我们现在需要添加一个能改变学生分数的函数，完整的代码实现如下：

```cpp
#include <vector>
#include <iostream>
#include <string>

class Student{
    private:
        // 成员变量
        char *name_;
        int age_;
        float score_;

    public:
        // 成员函数
        Student(char *name, int age, float score):
        	    name_(name), age_(age), score_(score){};
        void say() const;
        Student &setScore(float score);
};

void Student::say() const{
    std::cout << this->name_ << "的年龄是"<< this->age_ <<"，成绩是" << this->score_ << std::endl;
}

Student &Student::setScore(float score){
    this-> score_ = score;
    return *this;                             // 将 this 对象作为左值返回
}

int main(){
    Student *pstu = new Student("李华", 23, 87.5f);
    pstu->say();
    pstu->setScore(96.5f).say();
}
```

输出结果为：

```
╭─yangyun@yangyundeMBP.lan ~ ‹system› 
╰─➤  ./a.out 
李华的年龄是23，成绩是87.5
李华的年龄是23，成绩是96.5
```

这里的 `setScore` 成员函数的返回值是调用 `setScore` 的对象的引用。返回引用的函数是左值的，**这意味着这些函数返回的是对象本身而非对象的副本**。如果我们把一系列这样的操作连接在一条表达式的话：

```cpp
pstu->setScore(96.5f).say();
```

这些操作将在同一个对象上执行。在上面的表达式中，我们首先重置学生的分数，然后打印出结果。也就是说，上述语句等价于：

```cpp
pstu->setScore(96.5f);
pstu->say();
```

构造函数再探
------------
对于任何 C++ 的类来说，构造函数都是其中重要的组成部分。我们已经在 7.2.1 节中介绍了构造函数的基础知识，本节将继续介绍构造函数的一些其他功能，并对之前已经介绍的内容进行一些更深入的讨论。

### 构造函数初始值列表
一般来说，构造函数有两种初始化赋值的方式：

- 直接初始化数据成员 (效率更高）

```cpp
class Student{
    ...
    public:
        Student(char *name, int age, float score): name_(name), age_(age), score_(score){};
};
```
- 先初始化再赋值

```cpp
class Student{
    ...
    public:
        Student(char *name, int age, float score){
            name_ = name;
            age_  = age;
            score_= score;
        };
};
```
### 委托构造函数
C++11 新标准扩展了构造函数初始值的功能，使得我们可以定义所谓的委托构造函数（delegating constructor)。**一个委托构造函数使用它所属的类的其他构造函数执行它自己的初始化过程，或者说它把它自己的一些或者全部职责委托给了其他构造函数**。

举个例子：

```cpp
class Sales_data{
    public:
        // 非委托构造函数使用对应的实参初始化成员
        Sales_data(std::string s, unsigned cnt, double price):
            bookNo(s), units_sold(cnt), revenue(cnt*price) {}
        
        //  其余构造函数全都委托给第一个构造函数
        Sales_data(): Sales_data("", 0, 0) {}
        Sales_data(std::string s): Sales_data(s, 0, 0) {}
        Sales_data(std::istream &is): Sales_data(){ read(is, *this); }
};
```

在这个 Sales_data 类中，除了第一个构造函数以外其他三个都委托了它们的工作。第一个构造函数接受三个实参，使用这些实参初始化数据成员，然后结束工作。我们定义默认构造函数令其使用三个参数的构造函数来完成初始化过程，它也无须执行其他任务，这一点从空构造函数体能看得出来。
### explict 构造函数
C++ 中的 explicit 关键字用来修饰类的构造函数，表明该构造函数是显式的，既然有"显式"那么必然就有"隐式"，那么什么是显示而什么又是隐式的呢？

例如，我们构造了一个这样的类：

```cpp
class MyClass{
    private:
        int val_;
    public:
        MyClass(int val): val_(val) {}
        void add(MyClass m){ val_ += m.val_; }
};
```

然后我们这样构造了一个 `MyClass` 类：

```cpp
MyClass m = 10;       // 隐式转换，convert int to MyClass
m.add(5);             // 同上
```

在上面两行操作中，编译器默认都执行了一次隐式转换。例如，`add` 函数的形参本来应该是一个 `MyClass` 对象，但是这里它把 `int` 类型的 5 作为实参，实例化了一个 `MyClass` 类的临时对象再作为实参传给 `add` 函数。

隐式转换总是在我们没有察觉的情况下悄悄发生，除非心有所为，隐式转换常常是我们所不希望发生的。如果要避免这种自动转换的功能，我们该怎么做呢？嘿嘿这就是关键字 explicit 的作用了，将类的构造函数声明为"显式"（ 即在构造函数前面加上 explicit )，这样就可以防止这种自动的转换操作啦。

```cpp
explicit MyClass(int val): val_(val) {}
```

构造函数加上 explicit 关键字之后，之前那两种隐式转换的操作就是违法的了。通过将构造函数声明为 explicit（显式）的方式可以抑制隐式转换。**也就是说，explicit 构造函数必须显式调用**。

类的静态成员
------------
有的时候类需要它的一些成员与类本身直接相关，而不是与类的各个对象保持关联。例如，一个银行账户类可能需要一个数据成员来表示当前的基准利率。在此例中，我们希望利率与类关联，而非与每个类对象关联。从实现效率的角度来看，没必要每个对象都存储利率信息。而且更重要的是，一旦利率浮动，我们希望所有的对象都能使用新值。

- **声明静态成员**

我们通过在成员的声明之前加上关键字 static 使得其与类关联在一起。和其他成员一样，静态成员可以是 public 的或 private 的。静态数据成员的类型可以是常量，引用，指针等等。

举个例子，我们定义一个类，用它表示银行的账户记录：

```cpp
class Account{
    private:
        std::string owner;
        double amount;
        static double interest_rate;
        static double initRate();

    public:
        void calculate(){ amount += amount * interest_rate; }
        static double rate() {return interest_rate;}
        static void rate(double);
};
```

**类的静态成员存在于任何对象之外，对象中不包含任何与静态数据成员有关的数据**。因此，每个 `Account` 对象将包含两个数据成员：owner 和 amount。只存在一个 interest_rate 对象而且它被所有 `Account` 对象共享。

类似的，静态成员函数也不与任何对象绑定在一起，它们不包含 `this` 指针。作为结果，静态成员函数不能声明 const 的，而且**我们也不能在 static 函数体内使用 this 指针**。

- **使用静态成员**

我们使用作用域运算符能直接访问静态成员：

```cpp
double r;
r = Account::rate();             // 使用作用域运算符访问静态成员
```

**虽然静态成员不属于类的某个对象，但是我们仍然可以使用类的对象、引用或者指针来访问静态成员**：

```cpp
Account ac1;
Account *ac2 = &ac1;

r = ac1.rate();
r = ac2->rate();
```

第八章 文件的输入输出
===========
文本文件和二进制文件
---------------------------
从文件编码的方式来看，文件可分为 ASCII 码文件和二进制文件两种。ASCII 文件也称为文本文件，**这种文件在磁盘中存放时每个字符对应一个字节，用于存放对应的 ASCII 码**。二进制文件则是按二进制的编码方式来存放文件的。例如以浮点数 `0.375` 为对象，使用这两种格式对其存储：

- 小数 `0.375` 一共有 5 个字符：`'0'`、`'.'`、`'3'`、`'7'` 和 `'5'`，它们的 ASCII 码分别对应为 `48`、`46`、`51`、`55` 和 `53`，则存储的二进制流为：`00110000` `00101110` `00110011` `00110111` `00110111`。
- 二进制文件则是根据数据结构存储，它既可以存储浮点数，也可以存储整数、字符甚至结构体。浮点数 `0.375` 需要 4 个字节存储，也就是需要32位的宽度。如果我们将它转化成二进制编码，则为：`00111110110000000000000000000000`。

从上面的内容来看：文本文件是以字符 (char) 结构来存储， 而二进制文件不仅能以 char 存储，还能以 int，float，double 甚至 struct 的类型来存储。因此可以说，**文本文件只不过是一种特殊的二进制文件罢了**。要想打开二进制文件，就必须要知道该文件所对应的编码规范。例如对于文本文件，则可以用 Notepad++ 打开，对于 bmp 文件则需要图像查看器。


文件的读写操作
---------------------------

每种格式都有自己的优点。文本格式便于读取，可以使用文件编辑器来修改文本文件。二进制格式对于数字来说比较精确，因为它存储的是值的内部表示，因此不会有转换误差或舍入误差。以二进制格式保存数据的速度更快，因为不需要转换，并且通常占用的空间较小。

### 写文件

- 文本格式
来看一个具体的例子，考虑下面的结构定义和声明：

```cpp
struct student{
    char name[20];
    int age;
};
student stu = {"Yang Yun", 26};
```
要将结构 stu 的内容以文本格式保存，可以这样做：

```cpp
int main(){
    std::fstream fout("text.dat", std::ios::out);
    fout << stu.name << " " << stu.age << "\n";
    fout.close()
}
```
必须使用成员运算符显式地提供每个结构成员，还必须将相邻的数据分隔开，以便区分。如果结构有 30 个成员，则这项工作将很乏味。我们使用 vim 打开 text.dat 后：

```bashrc
Yang Yun 26
```

- 二进制格式
要用二进制格式存储相同的信息，可以这样做：

```cpp
int main(){
    std::fstream fout("binary.dat", std::ios::out | std::ios::binary);
    fout.write((char *) &stu, sizeof(stu));     
    fout.close();
}
```
上述代码使用计算机的内部数据表示，将整个结构作为一个整体保存。不能将该文件作为文本读取，但与文本相比，信息的保存更为紧凑、精确。它相对于文本格式保存，主要做了两个修改：

- 在原有基础上添加了二进制模式，`std::ios::binary`；
- 不再使用 `<<` 写入数据，而是使用了 `ostream& write (const char* s, streamsize n)` 成员函数；

write 函数的第 1 个参数为指向 char 型的常量指针，它是**内存的首地址**（由于 stu 的类型是结构体，因此我们需要它的地址强制转换成 char* 指针)；第 2 个参数是**整块内存的长度**，因此我们使用了 sizeof 运算符进行计算。

> 有个问题： 该程序是否可以使用 string 对象或者 char* 指针存储 student 的成员 name ？ 答案是否定的，问题在于 **string 对象本身实际上并没有包含字符串，而是包含了一个指向存储了字符串的内存单元的指针**。同理，char* 指针也是这样的。因此，文件并没保存字符串的内容，而是它的地址。下次你用程序读取该文件时，**这段地址是没有意义的**。

### 读文件

- 文本格式
前面提到过，文本文件是一个字符一个字符地存储。但是在读取文本文件时，我们可以直接使用 **getline** 函数逐行进行读取。依旧以之前生成的 text.dat 文件为例，程序如下：

```cpp
#include <fstream>
#include <iostream>

int main() {
    char *images_file = "text.dat";
    std::fstream fin(images_file, std::ios::in);

    if (fin.is_open()) {
        std::string line;
        while (std::getline(fin, line)) {       // 一行一行地读取打印
            std::cout << line << std::endl;
        }
        fin.close();
    } else {
        std::cout << "Unable to open " << images_file << std::endl;
    }
}
```

- 二进制格式
C++ 通过成员函数 `istream& read (char* s, streamsize n)` 可以读取二进制文件的内容，并且该函数的两个形参意义与 write 函数的相同，它的使用方法如下：

```cpp
int main(){
    student stu;
    std::fstream fin("binary.dat", std::ios::in | std::ios::binary);

    auto s = reinterpret_cast<char*>(&stu);
    fin.read(s, sizeof(student));

    std::cout << stu.name << " " << stu.age << std::endl;
}
```

上面程序是假设我们已经获得了结构体 student 声明的情况下发生的。那么**如果我们什么声明都没获得，在只有一个二进制文件 `binary.dat` 的情况下**，能不能读取它的内容呢？答案是肯定的，但是我们将无法解码它的内容，从而无法获得像 name 和 age 这样具体的信息。

```cpp
int main(){
    std::fstream fin("binary.dat", std::ios::in | std::ios::binary);

    fin.seekg(0, std::ios::end);
    int size = fin.tellg();                // 获取文件内存的字节长度

    auto str = (char *)malloc(size);
    fin.seekg(0, std::ios::beg);
    fin.read(str, size);                  // 读取内容，并复制内容给 str
}
```

seekp 和 seekg 函数用法
---------------------------

文件流对象有两个成员函数，分别是 [seekp](http://www.cplusplus.com/reference/ostream/ostream/seekp/) 和 [seekg](http://www.cplusplus.com/reference/istream/istream/seekg/)。它们可以用于将读写位置移动到文件中的任何字节。seekp 函数用于已经打开要进行写入的文件，而 seekg 函数则用于已经打开要进行读取的文件。可以将 "p" 理解为 "put"，将 "g" 理解为 "get"。

|语 句|如何影响读/写位置|
|---|---|
|file.seekp(32L, ios::beg);|将写入位置设置为从文件开头开始的第 33 个字节|
|file.seekp(-10L, ios::end);|将写入位置设置为从文件末尾开始的第 11 个字节|
|file.seekp(120L, ios::cur);|将写入位置设置为从当前位置开始的第 121 个字节|
|file.seekg(2L, ios::beg);|将读取位置设置为从文件开头开始的第 3 个字节|
|file.seekg(-100L, ios::end);|将读取位置设置为从文件末尾开始的第 101 个字节|
|file.seekg(0L, ios:beg);|将读取位置设置为文件开头|
|file.seekg(0L, ios:end);|将读取位置设置为文件末尾|

我们可以利用 seekg 函数来计算二进制文件的大小：首先将利用读取的光标位置放在文件的末尾，然后使用 [tellg](http://www.cplusplus.com/reference/istream/istream/tellg/) 函数返回当前字符的位置，也就文件内存的长度，以字节为单位。

```cpp
std::fstream fin("binary.dat", std::ios::in | std::ios::binary);

fin.seekg(0, std::ios::end);
int size = fin.tellg();
```

第九章 顺序容器
===========
顺序容器为程序员提供了控制元素存储和访问顺序的能力。这种顺序不依赖于元素的值，而是与元素加入容器时的位置相对应。与之对应的，有序和无序关联容器，则根据关键字的值来存储元素。

顺序容器概述
---------------------------

- **vector**：可变大小数组，支持快速随机访问。在尾部之外的位置插入或删除元素可能很慢。
- **deque**：双端队列，支持快速随机访问。在头尾位置进行插入/删除速度很快。
- **list**：双向链表，只支持双向顺序访问。在 list 中任何位置进行插入/删除操作速度都很快。
- **forward_list**：单向链表，只支持单向顺序访问。在链表任何位置进行插入/删除速度都很快。
- **array**：固定大小数组，支持快速随机顺序访问，不能添加或删除元素。
- **string**：与 vector 相似的容器，但专门用于保存字符。随机访问快，在尾部插入/删除速度很快。

**优劣分析：**

string 和 vector 将元素保存在连续的内存空间中。由于元素是连续存储的，因此通过元素下标来计算地址是非常快速的。但是，在这两种容器的中间位置添加或者删除元素就会非常耗时，因为每次插入和删除元素后，都需要移动后面所有的元素。

list 和 forward_list 两个容器的设计目的是令容器任何位置的添加和删除操作都很快速。作为代价，这两个容器不支持元素的随机访问：为了访问一个元素，我们只能遍历整个容器。而且，与 vector、deque 和 array 相比，这两个容器的额外内存开销也很大。

**哪种容器？**

> 通常，使用 vector 是最好的选择，除非你有很好的理由选择其他容器。

顺序容器操作
---------------------------
### 添加元素
除了 array 外，所有标准容器都提供灵活的内存管理。在运行时可以动态添加或删除元素来改变容器大小。

- **push_back**

push_back 可以将一个元素追加至一个 vector 尾部，除 array 和 forward_list 之外，每个顺序容器（包含 string 类型）都支持 push_back。例如，下面的循环每次读取一个 string 到 word 中，然后追加到尾部：

```cpp
string word;
while (cin >> word)
    container.push_back(word);
```

对 push_back 的调用在 container  尾部创建了一个新元素，将 container 的 size 增大了 1。该元素的值为 word 的一个拷贝。container 的类型可以是 list、vector 或 deque。

- **insert**

insert 提供了更加一般的功能， 它允许我们在容器中任意位置插入 0 个或多个元素。 vector、deque、list 和 string 都支持 insert 成员。每个 insert 函数都接受一个迭代器作为第一个参数。迭代器指出了在容器中什么位置放置新元素。由于迭代器可以指向容器尾部之后不存在的元素的位置，**所以 insert 函数将元素插入到迭代器所指定的位置之前**。

```cpp
vector<string> svec;
svec.insert(svec.begin(), "hello!");   // 插入到 begin 之前
```

> 将元素插入到 vector、deque 和 string 中的任何位置都是合法的。然而，这样做可能很耗时。

- **emplace_back**

**当我们调用 push_back 函数时，我们将元素类的对象传递给它们，这些对象被拷贝到容器中。而当我们调用一个 emplace_back 函数时，则是将参数传递给元素类型的构造函数**，并使用这些参数直接在容器管理的内存空间中直接构造元素。

```cpp
vector<Sales_data> c;
c.emplace_back("978", 25, 15.99);
c.push_back(Sale_data("978", 25, 15.99));
```

这两个方法都会创建新的 Sales_data 对象。在调用 emplace_back 时，会在容器管理的内存空间中直接创建对象。而调用 push_back 则会先创建一个局部对象，然后再在容器管理的内存空间中创建 Sale_data 对象并将临时对象里的内容拷贝过来。

### 访问元素

- **访问成员函数返回的是引用**

在容器中访问元素的成员函数（即，front、back、下标和 at）返回的都是引用。如果容器是一个 const 对象，则返回值是 const 的引用。如果容器不是 const 的，则返回值是普通引用，我们可以用来改变元素的值。

```cpp
#include <vector>

int main(){
    std::vector<int> c = {10, 12};

    if (!c.empty()){
        c.front() = 42;             // c = {42, 12}
        auto &v = c.back();
        v = 1024;                   // c = {42, 1024}, 此时 v 是 c 最后一个元素的引用
        auto v2 = c.back();
        v2 = 0;                     // c = {42, 1024}, 此时 v2 是 c 最后一个元素的拷贝
    }
}
```

- **下标操作和安全的随机访问**

顺序容器可以提供 `[]` 提供元素的随机访问，这种访问的优点在于速度极快，但是不会对下标参数进行越界检查。使用 `.at` 成员函数可以进行越界检查，但是它的访问速度会下降很多。

```cpp
vector<string> svec;              // 空 vector
cout << svec[0];                  // 运行时错误，svec 中没有元素
cout << svec.at(0);               // 抛出一个 out of range 异常
```

### 删除元素

- **pop_back**

pop_back 成员函数删除容器的尾元素，并返回 void。如果你需要弹出元素的值，就必须在执行弹出操作之前保存它：

```cpp
std::vector<int> c = {10, 24, 26, 25};

while (!c.empty()){
	int b = c.back();           // 删除前取出最后一个元素
    c.pop_back();               // 删除最后一个元素
}
```

- **erase**

erase 成员函数可以从容器指定位置删除元素。我们可以删除由一个迭代器指定的单个元素，也可以删除右一对迭代器指定范围的所有元素。

```cpp
std::vector<int> c = {10, 24, 26, 25};

c.erase(c.begin());                   // 删除第一个元素，c = {24, 26, 25}
c.erase(c.end()-2, c.end());          // 删除最后2元素 ，c = {24}
```

- **clear**

clear 成员函数会清空 vector 中所有的元素，但是**并不会释放所占用的内存**。如果想真正地释放所删除的内存，应该在 clear 或 erase 后面接  **shrink_to_fit** 函数。

```cpp
std::vector<int> c(10, 42);                    
std::cout << c.size() << " " << c.capacity() << std::endl;       // 10 10
c.clear();
std::cout << c.size() << " " << c.capacity() << std::endl;       // 0  10

c.shrink_to_fit();
std::cout << c.size() << " " << c.capacity() << std::endl;       // 0  0
```

### 改变容器大小
我们可以使用 resize 来增大或缩小容器：如果当前 size 大于所要求的 size，容器后部的元素会被删除；如果当前 size 小于所要求的 size，则会将新元素添加到容器后部。

```cpp
std::vector<int> c(10, 42);       // 10 个 int，每个元素都是 42

c.resize(15);                     // 将 5 个默认值为 0 的元素添加到末尾
c.resize(5);                      // 从末尾删除 20 个元素
```

此外，resize 操作还支持接受一个可选的元素值参数，用来初始化添加到容器中的元素。

```cpp
std::vector<int> c(10, 42);
c.resize(15, -1);                     // 将 5 个值为 -1 的元素添加到末尾
```

我们可以更深入地了解一下 reisze 操作内部发生的事，不妨先打印容器 resize 后存储元素的首地址：

```
std::vector<int> c(10, 42);
std::cout << c.data() << std::endl;        // 0x7f8bd1c05930
c.resize(15);                              // 扩大
std::cout << c.data() << std::endl;        // 0x7f8bd1c05960
c.resize(5);                               // 缩小
std::cout << c.data() << std::endl;        // 0x7f8bd1c05960
```

我们发现：容器元素的首地址在缩小后没有改变，而在容器扩大后发生了变化。这是由于容器扩充后需要重新分配内存，并且还需要把重新移动所有元素，因此地址发生了改变。

vector 对象是如何增长的
---------------------------
为了支持快速随机访问， vector 将元素连续存储，即每个元素紧挨着前一个元素进行存储。假定容器中元素是连续存储的，且容器的大小可变。**考虑向 vector 添加元素，如果没有空间容纳新元素，那么容器只能重新申请一块更大的新空间，并且必须把已有元素从旧空间移动到新空间，然后添加新元素，释放旧空间。如果我们每添加一个元素，那么 vector 就得执行一次这样的内存分配和释放操作，性能就会慢到不可接受**。


为了尽量减少容器在添加元素时重新分配内存的次数，容器在每次分配内存时都会申请比新空间更大的内存空间。容器预留这些空间作为备用，可用来保存更多的新元素。这样就不需要每次添加元素都重新分配容器的内存空间了。

- **capacity 和 size 的区别**

理解 capacity 和 size 的区别非常重要。容器的 size 是指它已经保存元素的数目，而 capacity 则是在不分配新的内存空间的前提下最多可以保存多少个元素。

```cpp
std::vector<int> ivec;
std::cout << "ivec: size " << ivec.size()
          << " capacity "  << ivec.capacity() << std::endl;

for(int i=0; i<24; i++)
    ivec.push_back(i);

std::cout << "ivec: size " << ivec.size()
          << " capacity "  << ivec.capacity() << std::endl;
```

当我们的程序运行时，输出的结果如下所示：

```
ivec: size 0 capacity 0
ivec: size 24 capacity 32
```

我们知道一个 vector 的 size 为 0，显然在我们的标准库实现一个空 vector 的 capacity 也为 0。当向 vector 添加元素时，我们知道 size 与添加的元素数目相等。而 capacity 至少与 size 一样大，具体会分配多少额外空间则视标准库的具体实现而定。

可以想象 ivec 的当前状态如下图所示：

`0`|`1`|`2`|`3`|`...`|`23`|`预留空间`

- **容器怎么重新申请空间？**

前面讲到容器 vector 为了减少重新分配内存的次数，而会保留一些预留空间。因此出现 ivec 的 size=24，capacity=32 的情况，那么每次内存不够而需要重新申请空间的时候又是怎么进行的呢？依旧以前面的 push_back 过程举例，我们打印出容器每次添加元素后的容量：

```cpp
for(int i=0; i<24; i++){
    ivec.push_back(i);
    std::cout << "ivec: size " << ivec.size()
              << " capacity "  << ivec.capacity() << std::endl;
}
```

```
ivec: size 1 capacity 1
ivec: size 2 capacity 2          /* 重新申请空间 */
ivec: size 3 capacity 4          /* 重新申请空间 */
ivec: size 4 capacity 4
ivec: size 5 capacity 8
ivec: size 6 capacity 8
ivec: size 7 capacity 8
ivec: size 8 capacity 8
ivec: size 9 capacity 16         /* 重新申请空间 */
ivec: size 10 capacity 16
ivec: size 11 capacity 16
ivec: size 12 capacity 16
ivec: size 13 capacity 16
ivec: size 14 capacity 16
ivec: size 15 capacity 16
ivec: size 16 capacity 16
ivec: size 17 capacity 32        /* 重新申请空间 */
ivec: size 18 capacity 32
ivec: size 19 capacity 32
ivec: size 20 capacity 32
ivec: size 21 capacity 32
ivec: size 22 capacity 32
ivec: size 23 capacity 32
ivec: size 24 capacity 32 
```

我们发现：只要容器当前的 size < capacity 时，每次往容器内 push_back 元素时，都不会重新申请空间。但是如果 vector 的容量不够时，那么每次需要分配新内存时将当前的容量翻倍。

第十章 关联容器
===========
关联容器支持高效的关键字查找和访问。两个主要的关联容器类型是 map 和 set。其中 map 的元素是一些 key-value 对：关键字起到索引的作用，值则表示与索引相关联的数据。set 中每一个元素只包含一个关键字；set 支持高效的关键字查询操作，检查一个给定的关键字是否在 set 中。

使用关联容器
---------------------------
- **使用 map**

一个经典的使用关联容器来进行单词计数的程序：

```cpp
#include <map>
#include <iostream>

int main(){
    std::map<std::string, size_t> word_count;

    std::string word;
    while(std::cin>>word){
        word_count[word]++;
    }

    for(const auto &w: word_count){
        std::cout << w.first << ": " << w.second << std::endl;
    }
}
```

`while` 循环每次从标准输入读取一个单词。它使用每个单词对 `word_count` 进行下标操作。如果 `word` 还未在 `map` 中，下标运算符会创建一个新元素，其关键字为 `word`，值为 0. 不管元素是否是新创建的，我们将其值加 1.

一旦读取完所有的输入，范围 for 语句就会遍历 map，打印每个单词和对应的计数器。当从 map 中提取一个元素时，会得到一个 **pair** 类型的对象。`map` 所使用的 `pair` 用 `first` 成员保存关键字，用 `second` 保存对应的值。

- **使用 set**

我们可以使用 set 容器来判断输入的单词是否在集合里面：

```cpp
#include <set>
#include <iostream>

int main(){
    std::set<std::string> data = {"hello", "world", "but", "c++", "fuck"};
    std::string world;

    while(std::cin>>world)
        std::cout << world << ": " << (data.find(world) != data.end()) << std::endl;
}
```

我们发现，程序是通过以下方式判断单词是否在集合中：

```cpp
(data.find(world) != data.end())
```

**`find` 调用返回一个迭代器，如果给定关键字在 `set` 中，迭代器则指向该关键字。否则，`find` 返回尾后迭代器**。

关联容器概述
---------------------------
### 定义关联容器
- **初始化 map 和 set**

如前所示，当定义一个 map 时，必须既指明关键字类型又指明值类型；而定义一个 set 时，只需指明关键字类型。在 C++11 新标准下，我们也可以对关联容器进行值初始化：

```cpp
std::set<std::string> data = {"hello", "world", "but", "c++", "fuck"};
std::map<std::string, size_t> word_count = { {"hello", 2},
                                             {"word",  0},
                                             {"fuck",  1},
                                             {"but",   0},
                                             {"c++",   0}};
```

当初始化一个 map 时，必须提供关键字类型和值类型。我们将每个关键字-值对包围在花括号中：

```
{key, value}
```

来指出它们一起构成了 map 中的一个元素。**在每个花括号中，关键字是第一个元素，值是第二个元素**。

- **初始化 multimap 和 multiset**

一个 map 或 set 中的关键字必须是唯一的，即对于一个给定的关键字，只能有一个元素的关键字等于它。容器 multimap 和 multiset 没有此限制，它们都允许多个元素具有相同的关键字。

```cpp
std::multimap<char,int> first;

first.insert(std::pair<char,int>('a',10));
first.insert(std::pair<char,int>('b',15));
first.insert(std::pair<char,int>('b',20));
first.insert(std::pair<char,int>('c',25));

int myints[]= {10,20,30,20,20};
std::multiset<int> second (myints,myints+5);       // pointers used as iterators
```
### pair 类型
在介绍关联容器操作之前，我们需要了解名为 pair 的标准库类型。它定义在头文件 utility 中。一个 pair 对象只保存两个数据成员，其初始化方式如下:

```cpp
#include <utility>
#include <iostream>

int main(){
    std::pair<int, float> p1;
    std::cout << "Value-initialized: "
              << p1.first << ", " << p1.second << '\n';       // Value-initialized: 0, 0

    std::pair<int, double> p2(42, 0.123);
    std::cout << "Initialized with two values: "
              << p2.first << ", " << p2.second << '\n';       // Initialized with two values: 42, 0.123
}
```

与其他标准库类型不同，pair 的数据成员是 public 的。两个成员分别命名为 first 和 second，可以直接访问。除了上面两种创建方式以外，我们还可以通过 **make_pair** 函数创建：

```cpp
auto p3 = std::make_pair(3, 3.14);
```
关联容器操作
---------------------------
### 关联容器迭代器

- **map 的迭代器**

我们可以获得 map 中一个元素的迭代器，并通过该迭代器修改元素对应的值：

```cpp
std::map<std::string, size_t> word_count = { {"hello", 2},
                                             {"word",  0}};
                                             
auto map_it = word_count.begin();      // *map_it 是指向 pair<const string, size_t> 对象的引用
std::cout << map_it->first << ": " << map_it->second << std::endl;     // hello: 2

++map_it->second;                      // 修改元素的 value
std::cout << map_it->first << ": " << map_it->second << std::endl;     // hello: 3
```

- **set 的迭代器**

set 也提供了迭代器来访问元素，但与 map 不同的是：它允许只读性质地访问 set 中的元素，并不能改变它的值。

```cpp
std::set<std::string> data = {"hello", "world", "but", "c++", "fuck"};
for(auto it = data.begin(); it != data.end(); it++){
    // *it = "yun";   // 错误 ！ set 中的关键字是只读的
    std::cout << *it << std::endl;
}
```

### 添加元素
对一个 map 进行 insert 操作时，必须记住元素类型是 pair。通常，对于想要插入的数据，并没有一个现成的 pair 对象。可以在 insert 的参数列表中创建一个 pair：

```cpp
std::map<std::string, size_t> word_count = { {"hello", 2} };

word_count.insert({"world", 1});
word_count.insert(std::make_pair("but", 2));
word_count.insert(std::pair<std::string, size_t>("huya", 1));
word_count.insert(std::map<std::string, size_t>::value_type("next", 3));
```

### 删除元素

```cpp
// 1. 删除键为 "bfff" 指向的元素
cmap.erase("bfff");
 
// 2. 删除迭代器 key 所指向的元素
map<string,int>::iterator key = cmap.find("mykey");
if(key!=cmap.end()){
    cmap.erase(key);
}
 
// 3. 删除所有元素
cmap.erase(cmap.begin(),cmap.end());
```
