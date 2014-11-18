#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
//#include "prettyprint.hpp"

using namespace std;

int main(int argc, char* argv[]){
  double threshold=0.97;
  if(argv[2]) threshold=stod(argv[2]);
  ifstream ifs {argv[1]};
  vector<vector<string>> table;
  for(string line; getline(ifs,line);){
    stringstream ssline(line);
    vector<string> row;
    for(string cell; getline(ssline,cell,',');){
      row.push_back(cell);
    }
    table.push_back(row);
  }
  //cout << table.size() << ", " << table.at(0).size() << endl;
  vector<size_t> exclude_columns;
  for(size_t row; row<table.size();++row){
    vector<string> matched;
    matched.push_back(table[row].front());
    vector<int> correlated;
    for(size_t i=row; i<table[row].size();++i){
      if(find(begin(exclude_columns),end(exclude_columns),i)==exclude_columns.end()){
        if(abs(strtod(table[row][i].c_str(),nullptr))>threshold){
          matched.push_back(table[row][i]);
          if(i>row)
            exclude_columns.push_back(i);
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
    //bool after1=1;
    //for(auto cell:row){
      //if(strtod(cell.c_str(),nullptr)==1.0) after1=1;
      //if(after1) if(strtod(cell.c_str(),nullptr)>threshold){
      //if(strtod(cell.c_str(),nullptr)>threshold){
        //matched.push_back(cell);
      //}
    //}
    if(matched.size()>2) {
      for(auto i:matched) cout << i << " ";
      cout << endl;
    }
  }
  cout << "total " << exclude_columns.size()<< endl;
  for(auto i : exclude_columns) cout << i << " ";
  cout << endl;
  //cout << exclude_columns<< endl;
  return 0;
}
