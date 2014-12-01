#include "headers.hpp"

using namespace std;

int main(int argc, char* argv []) {
  ifstream ifs(argv[1]);
  for(string line; getline(ifs,line);){
    cout << stoul(line)+1 << endl;
  }
  return 1;
}

