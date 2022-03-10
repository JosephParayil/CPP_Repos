#include <iostream>
#include <vector>
using namespace std;

struct student {
    int a;
    int b;
};


void solve() {
    int n, m; cin >> n >> m;
    vector<student> students(n,{-1,-1});
    int a,b;
    while (m--) {
        cin >> a >> b;

        a--;b--;
        int* aHand;
        int* bHand;

        if (students[a].a==-1) aHand = &students[a].a;
        else if (students[a].b==-1) aHand = &students[a].b;
        else continue;
        if (students[b].a==-1) bHand = &students[b].a;
        else if (students[a].b==-1) bHand = &students[b].b;
        else continue;
        
        *aHand = b;
        *bHand = a;
    }
    
    vector<int> explored;
    int largest = 0;
    for (int i=0;i<n;i++) {
        
        bool found = false;
        for (int j=0;j<explored.size();j++) {
            if (explored[j]==i) { found = true; break;}
        }
        if (found) continue;
        int count = 1;
        vector<int> local = {i};
        explored.push_back(i);
        vector<int> stack = {i};
        vector<bool> dirStack = {false};
        int l;
        while (stack.size()) {
            l = stack.size()-1;
            found = false;
            if (dirStack[l]) {
                if (students[stack[l]].b==-1) {
                    stack.pop_back(); dirStack.pop_back();
                    continue;
                }
                for (int j=0;j<local.size();j++) {
                    if (local[j]==students[stack[l]].b) {found = true; break;}
                }
                if (found) {stack.pop_back(); dirStack.pop_back(); continue;}
                count++;
                local.push_back(students[stack[l]].b);
                explored.push_back(students[stack[l]].b);
                stack.push_back(students[stack[l]].b);
                dirStack.push_back(false);
            } else {
                if (students[stack[l]].a==-1) {
                    stack.pop_back(); dirStack.pop_back();
                    continue;
                }
                for (int j=0;j<local.size();j++) {
                    if (local[j]==students[stack[l]].a) {found = true; break;}
                }
                if (found) {dirStack[l]=true; continue;}
                count++;
                local.push_back(students[stack[l]].a);
                explored.push_back(students[stack[l]].a);
                stack.push_back(students[stack[l]].a);
                dirStack.push_back(false);
            }
            dirStack[l]=true;
        }
        if (count>largest) largest=count;
    }
    cout << largest << '\n';
    
}

int main() {
    int t; cin>>t;
    while(t--) solve();
    return 0;
}