#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
//#include "prettyprint.hpp"

using namespace std;

vector<vector<string>> loadtxt(char filename[]){
  vector<vector<string>> table;
  ifstream ifs {filename};
  for(string line; getline(ifs,line);){
    stringstream ssline(line);
    vector<string> row;
    for(string cell; getline(ssline,cell,',');){
      row.push_back(cell);
    }
    table.push_back(row);
  }
  //cout << table.size() << ", " << table.at(0).size() << endl;
  return table;
}

auto columns_over_threshold(const vector<vector<string>> &table, double threshold) {
  vector<size_t> exclude_columns;
  vector<string> exclude_columns_name;
  for(size_t row=0; row<table.size();++row){
    vector<string> matched;
    matched.push_back(table[row].front());
    vector<int> correlated;
    for(size_t i=row; i<table[row].size();++i){
      if(find(begin(exclude_columns),end(exclude_columns),i)==exclude_columns.end()){
        if(abs(strtod(table[row][i].c_str(),nullptr))>threshold){
          matched.push_back(table[row][i]);
          if(i>row)
            exclude_columns.push_back(i);
            exclude_columns_name.push_back(table.front()[i]);
          correlated.push_back(i);
        }
      }
    }
    if(correlated.size()>1){
      cout << correlated.size()-1 << " ";
      for(auto i:correlated) cout << i << " " << table.front()[i]<<" ";
      //cout << endl;
      cout << " -- ";
    }
    if(matched.size()>2) {
      for(auto i:matched) cout << i << " ";
      cout << endl;
    }
  }
  //return {exclude_columns,exclude_columns_name};
  return make_tuple(exclude_columns,exclude_columns_name);
}

int main(int argc, char* argv[]){
  double threshold=0.97;
  if(argv[2]) threshold=stod(argv[2]);
  auto table = loadtxt(argv[1]);
  auto exclude_columns = columns_over_threshold(table,threshold);

  cout << "total " << get<1>(exclude_columns).size()<< endl;
  for(auto i : get<1>(exclude_columns)) cout << i << " ";
  cout << endl;
  return 0;
}
