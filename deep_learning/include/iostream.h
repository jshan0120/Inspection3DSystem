#include <iostream>
#include <fstream>
#include <string>

class IOStream {
public:
    IOStream(const std::string& path) {
        f.open(path, std::ios::out | std::ios::app);
        if (!f.is_open()) {
            std::cerr << "Error: Could not open file " << path << std::endl;
        }
    }

    void cprint(const std::string& text) {
        std::cout << text << std::endl;
        
        if (f.is_open()) {
            f << text << std::endl;
            f.flush();
        }
    }

    void close() {
        if (f.is_open()) {
            f.close();
        }
    }

    ~IOStream() {
        close();
    }

private:
    std::ofstream f;
};