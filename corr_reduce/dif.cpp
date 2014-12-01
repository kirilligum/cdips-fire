#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
  ifstream ifs{argv[1]};
  double old=0;
  vector<double> dif;
  vector<double> vals;
  for(string line; getline(ifs,line);){
    stringstream iss(line);
    string a;
    getline(iss,a,' ');
    getline(iss,a,' ');
    double val = stod(a);
    vals.push_back(val);
    dif.push_back(val-old);
    old =val;
  }
  for(auto i: vals) cout << i << ' ';
  cout << endl;
  for(auto i: dif) cout << i << ' ';
  cout << endl;
  return 1;
}
