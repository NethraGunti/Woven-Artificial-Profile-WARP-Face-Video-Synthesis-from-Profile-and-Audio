def getMax(arr):
	mid = (len(arr)-1)//2 # 4
	start = 0
	end = len(arr) - 1 # 9

	while start <= end:
		if arr[mid] > arr[mid-1] and arr[mid] > arr[mid+1]:
			return arr[mid]
		elif arr[mid] > arr[mid-1] and arr[mid] <= arr[mid+1]:
			start = mid+1 # 5
			mid = (start + end)//2 # 7
		elif arr[mid] >= arr[mid-1] and arr[mid] <= arr[mid+1]:
			end = mid - 1 #6
			mid = (start + end)//2 
		elif  end - start == 1:
			return arr[mid]

arr = [1, 2, 3, 3, 4, 4, 4, 4, 4, 1]
print(getMax(arr))