#include <iostream>
#include <regex>
using namespace std;

void solve() {
    string line; getline(cin,line);
    cout << regex_replace(line,regex("e"),"") << '\n';
}

int main() {
    int t; cin>>t;
    string input; getline(cin,input);
    while(t--) solve();
    return 0;
}