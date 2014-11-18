#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[]) {
  std::fstream f{argv[1]};
  std::vector<double> v;
  for(double x; f>>x;)
    v.push_back(x);
  sort(begin(v),end(v));
  for(auto i : v) cout << i << " ";
  cout << endl;
  return 0;
}

