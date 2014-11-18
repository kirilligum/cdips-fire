#include <sstream>
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;

int main (int argc, char * argv[]){
  vector<string> vs {"","0.0","zz","NaN","12.3"};
  for(string s: vs) {
    char * p;
    double d = strtod(s.c_str(),&p);
    if(*p) cout << "wrong\n";
    cout << s << endl;
    cout << d << endl;
  }
  return 0;
}

